import cv2
from deepface import DeepFace

def run_race_detection(source_video_path):
    cap = cv2.VideoCapture(source_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = DeepFace.analyze(frame_rgb, actions=['race'], enforce_detection=False)
        except Exception as e:
            print("Detection failed:", e)
            results = []

        if results and (isinstance(results, dict) or isinstance(results, list)):
            if isinstance(results, dict):
                results = [results]

            height, width = frame.shape[:2]
            for face in results:
                region = face.get('region', {})
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)

                dominant_race = face.get('dominant_race', 'Unknown')

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_race, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Race Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
