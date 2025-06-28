from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import numpy as np

classes = ['cloudy', 'rainy', 'shine', 'sunrise']

def preprocess_frame(frame):
    frame = cv2.resize(frame, (100, 100))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    return frame

def run_weather_detection(source_video_path):
    model = load_model("modules/weather/trainedModelE40.h5")
    cap = cv2.VideoCapture(source_video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_img = preprocess_frame(frame)
        pred = model.predict(input_img)
        label = classes[np.argmax(pred)]

        cv2.putText(frame, f"Weather: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Weather Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
