import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import math
from collections import deque

class PosePhoneDetector:
    def __init__(self,
                 model_path="best4.pt",
                 detection_interval=3,
                 conf_threshold=0.25,
                 iou_threshold=0.45,
                 min_area=0.0002,     # نسبت نرمال شده به فریم
                 max_area=0.9,
                 aspect_min=0.1,
                 aspect_max=4,
                 require_hand_confirmation=False,  # اگر True: فقط وقتی دست نزدیکه قبول کن
                 hand_dist_thresh=0.20,  # آستانه نزدیکی (نسبت نرمال به بزرگ‌ترین بعد فریم)
                 persistence_frames=2):  # تشخیص باید در این تعداد detect-frame دیده بشه
        # pose + hands
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # yolov8 model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model '{model_path}': {e}")
            raise

        # پارامترهای inference / filtering
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_detections = []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.require_hand_confirmation = require_hand_confirmation
        self.hand_dist_thresh = hand_dist_thresh
        self.persistence_frames = persistence_frames

        # برای persistence (حفظ ثبات تشخیص)
        # نگه می‌داره تاریخچه‌ی مراکز باکس‌ها (deque از طول persistence_frames)
        self.history = deque(maxlen=self.persistence_frames)

        # نوک انگشت‌ها برای Mediapipe hands
        self.fingertip_ids = [4, 8, 12, 16, 20]

    # ---------- helper ----------
    def iou(self, boxA, boxB):
        # box = (x1,y1,x2,y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
        boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
        union = boxAArea + boxBArea - interArea + 1e-9
        return interArea / union

    def box_center(self, box):
        x1,y1,x2,y2 = box
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    # آیا Fingertip یا wrist نزدیک مرکز باکس هست؟
    def hand_confirms_box(self, box, hand_landmarks, frame_w, frame_h):
        # اگر hand_landmarks None -> False
        if not hand_landmarks:
            return False

        cx, cy = self.box_center(box)
        # چک کن نوک انگشت داخل باکس باشه یا فاصله کمتر از threshold
        for hand in hand_landmarks:
            # محاسبه مختصات نوک انگشت‌ها
            for fid in self.fingertip_ids:
                lm = hand.landmark[fid]
                fx, fy = lm.x * frame_w, lm.y * frame_h
                # داخل باکس؟
                if (fx >= box[0] and fx <= box[2] and fy >= box[1] and fy <= box[3]):
                    return True
                # یا فاصله مرکز باکس تا نوک انگشت کم باشه (نسبت به بزرگ‌ترین بعد فریم)
                dist = math.hypot((fx - cx), (fy - cy))
                if dist <= self.hand_dist_thresh * max(frame_w, frame_h):
                    return True

            # همچنین میتوانیم مچ دست را بررسی کنیم (landmark 0)
            wrist = hand.landmark[0]
            wx, wy = wrist.x * frame_w, wrist.y * frame_h
            dist_w = math.hypot((wx - cx), (wy - cy))
            if dist_w <= self.hand_dist_thresh * max(frame_w, frame_h):
                return True

        return False

    # ---------- تشخیص باکس‌ها (با فیلتر اولیه) ----------
    def detect_phone(self, frame):
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            # همچنان از last_detections استفاده کن
            return self.last_detections

        h, w = frame.shape[:2]

        # inference با آستانه conf اولیه
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                # گرفتن مختصات و confidence
                xy = box.xyxy[0].cpu().numpy()   # [x1,y1,x2,y2]
                try:
                    conf = float(box.conf.cpu().numpy())
                except:
                    conf = float(box.conf) if hasattr(box, "conf") else 0.0

                cls_id = int(box.cls)

                # ---- اصلاح شده: فقط کلاس شماره 0 (phone) را قبول کن ----
                if cls_id != 0:
                    continue

                x1,y1,x2,y2 = xy
                bw = max(1, x2-x1); bh = max(1, y2-y1)
                area_norm = (bw * bh) / (w * h)
                aspect = bw / (bh + 1e-9)

                # فیلتر اندازه و نسبت
                if area_norm < self.min_area or area_norm > self.max_area:
                    continue
                if aspect < self.aspect_min or aspect > self.aspect_max:
                    continue
                # فیلتر بر اساس confidence دوباره (اینجا اختیاریه)
                if conf < self.conf_threshold:
                    continue

                detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        # persistence: نگه داشتن تاریخچه مراکز و افزایش اعتماد برای باکس‌های مکرر
        # ذخیره مراکز جدید
        centers = [self.box_center(box[:4]) for box in detections]
        self.history.append(centers)

        # اگر persistence_frames>1، فقط آن باکس‌هایی را قبول کن که در حداقل یکسان در چند فریم دیده شده‌اند
        if self.persistence_frames > 1 and len(self.history) == self.persistence_frames:
            kept = []
            for i, box in enumerate(detections):
                bx = box[:4]
                matches = 0
                # چک کن در هر فریم اخیر حداقل یک مرکز با IoU بالا وجود داشته باشه
                for past_centers in self.history:
                    for pc in past_centers:
                        # تبدیل pc به باکس کوچک حول مرکز برای IoU چک ساده
                        px, py = pc
                        # باکس فرضی کوچک
                        eps = 20  # pixels tolerance
                        pbox = (px-eps, py-eps, px+eps, py+eps)
                        if self.iou(bx, pbox) > 0.2:
                            matches += 1
                            break
                if matches >= 1:  # حداقل در یکی از فریم‌ها دیده شده
                    kept.append(box)
            detections = kept

        self.last_detections = detections
        return detections

    # ---------- بررسی با دست و نمایش نهایی ----------
    def analyze_pose(self, frame, phone_boxes):
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_pose = self.pose.process(image_rgb)
        results_hands = self.hands.process(image_rgb)

        # رسم اسکلت مثل کد اصلی (اگر موجود)
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            shoulders = {
                "left": [lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                         lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y],
                "right": [lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                          lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y]
            }
            elbows = {
                "left": [lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x,
                         lm[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y],
                "right": [lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x,
                          lm[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y]
            }
            wrists = {
                "left": [lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                         lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y],
                "right": [lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                          lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y]
            }
            # رسم خطوط شانه-آرنج-مچ
            for side, color in zip(["left", "right"], [(255, 200, 100), (100, 200, 255)]):
                pts = [shoulders[side], elbows[side], wrists[side]]
                for i, pt in enumerate(pts):
                    cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 8, color, -1)
                    if i > 0:
                        cv2.line(frame,
                                 (int(pts[i - 1][0] * w), int(pts[i - 1][1] * h)),
                                 (int(pts[i][0] * w), int(pts[i][1] * h)),
                                 color, 3)

        # رسم نوک انگشت‌ها و خطوط مچ->نوک (اگر دست‌ها پیدا شد)
        hand_landmarks = []
        if results_hands.multi_hand_landmarks:
            for hand_landmarks_single in results_hands.multi_hand_landmarks:
                hand_landmarks.append(hand_landmarks_single)
                # draw wrist & fingertips
                wrist = hand_landmarks_single.landmark[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.circle(frame, (wx, wy), 6, (0,255,255), -1)
                for fid in self.fingertip_ids:
                    lm_f = hand_landmarks_single.landmark[fid]
                    cx, cy = int(lm_f.x * w), int(lm_f.y * h)
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 255), -1)
                    cv2.line(frame, (wx, wy), (cx, cy), (200, 100, 255), 2)

        # فیلتر نهایی باکس‌ها با استفاده از دست (اختیاری)
        final_boxes = []
        for box in phone_boxes:
            x1,y1,x2,y2,conf = box
            accept = True

            # اگر require_hand_confirmation فعال است، حتماً یکی از شرایط دست را داشته باش
            if self.require_hand_confirmation:
                if not self.hand_confirms_box((x1,y1,x2,y2), hand_landmarks, w, h):
                    accept = False

            # رسم باکس‌های قبول شده
            if accept:
                final_boxes.append((x1,y1,x2,y2,conf))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
                cv2.putText(frame, f"phone {conf:.2f}", (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        status = "No Phone Detected" if not final_boxes else "Phone Detected"
        return frame, status

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Couldn't open source:", source)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            phone_boxes = self.detect_phone(frame)
            frame, status = self.analyze_pose(frame, phone_boxes)

            # اصلاح نمایش وضعیت: پایدار و واضح
            color = (0,255,0) if status == "Phone Detected" else (0,0,255)
            cv2.putText(frame, status, (30,50), cv2.FONT_HERSHEY_COMPLEX, 1.0, color, 2)
            cv2.imshow("Phone + Pose", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# مثال اجرا
if __name__ == "__main__":
    detector = PosePhoneDetector(
        model_path="best4.pt",
        conf_threshold=0.25,
        min_area=0.0002,
        max_area=0.8,
        aspect_min=0.1,
        aspect_max=4.0,
        detection_interval=1,    # هر فریم detect کن
        persistence_frames=1,    # بدون سخت‌گیری اضافی
        require_hand_confirmation=False
    )
    detector.run(source=0)