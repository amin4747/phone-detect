import cv2
from ultralytics import YOLO

class PhoneDetector:
    def __init__(self, model_path='best3.pt'):
        self.model = YOLO(model_path)

    def run(self, source=0, save_output=False, output_path="output.mp4"):
        """
        source می‌تونه باشه:
        - عدد (0 یا 1) برای وبکم یا دوربین
        - آدرس RTSP یا HTTP برای دوربین مدار بسته
        - مسیر فایل ویدیو مثل "video.mp4"
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print('Error')
            return
        else:
            print("camera's open")

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=0.3, verbose=False)

            annotated_frame = results[0].plot()

            cv2.imshow("Phone Detection", annotated_frame)

            if writer:
                writer.write(annotated_frame)

            # کلید ESC = خروج
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    detector = PhoneDetector(model_path='best3.pt')

    # برای فایل ویدیو
    #detector.run(source="cctv.mp4", save_output=True, output_path="result.mp4")

    # برای وبکم / دوربین
    detector.run(source=0)

    # برای RTSP/HTTP استریم دوربین
    # detector.run(source="rtsp://user:pass@ip:port/stream")
    #fdg
