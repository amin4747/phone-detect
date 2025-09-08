import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

class PosePhoneDetector:
    def __init__(self, detection_interval=3):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.model = YOLO("yolov8m.pt")  
        self.detection_interval = detection_interval
        self.frame_count = 0
        self.last_detections = []

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(np.degrees(radians)) 
        return angle if angle <= 180 else 360 - angle

    def detect_phone(self, frame):
        self.frame_count += 1
        if self.frame_count % self.detection_interval != 0:
            return self.last_detections

        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = self.model.names[cls_id]
                if "phone" in label.lower():
                    detections.append(box.xyxy[0].cpu().numpy())
        self.last_detections = detections
        return detections

    def analyze_pose(self, frame, phone_boxes):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return frame, "No Pose"

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark

        nose = [lm[mp.solutions.pose.PoseLandmark.NOSE].x,
                lm[mp.solutions.pose.PoseLandmark.NOSE].y]
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

        # خطوط لندمارک: آبی ملایم
        for side, color in zip(["left", "right"], [(255, 200, 100), (100, 200, 255)]):
            pts = [shoulders[side], elbows[side], wrists[side]]
            for i, pt in enumerate(pts):
                cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 8, color, -1)
                if i > 0:
                    cv2.line(frame,
                             (int(pts[i - 1][0] * w), int(pts[i - 1][1] * h)),
                             (int(pts[i][0] * w), int(pts[i][1] * h)),
                             color, 3)

        if not phone_boxes:
            return frame, "No Phone Detected"

        for box in phone_boxes:
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            for side in ["left", "right"]:
                elbow_angle = self.calculate_angle(shoulders[side], elbows[side], wrists[side])
                wrist_face_dist = np.linalg.norm(np.array(wrists[side]) - np.array(nose))

                # شرط: گوشی + خم بودن یا نزدیکی دست به صورت
                if elbow_angle < 150 or wrist_face_dist < 0.35:
                    return frame, "Phone Use"

        return frame, "Not Using"

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            phone_boxes = self.detect_phone(frame)
            frame, status = self.analyze_pose(frame, phone_boxes)

            # فونت شیک و رنگ سبز برای استفاده
            font = cv2.FONT_HERSHEY_COMPLEX
            color = (0, 255, 0) if status == "Phone Use" else (0, 0, 255)
            cv2.putText(frame, status, (30, 50), font, 1.2, color, 3, cv2.LINE_AA)

            cv2.imshow("Pose & Phone Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    PosePhoneDetector().run()
