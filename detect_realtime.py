"""
Real-time facial emotion recognition using webcam.

Pipeline:
    1. Capture video frame from webcam
    2. Convert to grayscale
    3. Detect faces using Haar-cascade classifier
    4. For each face: crop, resize to 48×48, normalize, predict emotion
    5. Draw bounding box and emotion label on frame
    6. Display result — press 'q' to quit

Usage:
    python detect_realtime.py
    python detect_realtime.py --model model_file.keras --camera 0
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model


EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# Colors (BGR format)
BOX_COLOR = (50, 50, 255)      # Red-ish
LABEL_BG_COLOR = (50, 50, 255)
TEXT_COLOR = (255, 255, 255)    # White


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    """
    Preprocess a cropped grayscale face for the CNN.

    Steps:
        1. Resize to 48×48 (model input size)
        2. Normalize pixel values to [0, 1]
        3. Reshape to (1, 48, 48, 1) for batch prediction
    """
    resized = cv2.resize(gray_face, (48, 48))
    normalized = resized / 255.0
    return np.reshape(normalized, (1, 48, 48, 1))


def main():
    parser = argparse.ArgumentParser(description="Real-time emotion recognition")
    parser.add_argument("--model", type=str, default="model_file.keras",
                        help="Path to the trained Keras model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    # Load Haar-cascade face detector (bundled with OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load Haar-cascade classifier.")
        return

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}.")
        return

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            # Crop and preprocess face
            face_roi = gray[y : y + h, x : x + w]
            input_tensor = preprocess_face(face_roi)

            # Predict emotion
            predictions = model.predict(input_tensor, verbose=0)
            label_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][label_idx]
            emotion = EMOTION_LABELS[label_idx]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

            # Draw label background
            cv2.rectangle(frame, (x, y - 40), (x + w, y), LABEL_BG_COLOR, -1)

            # Draw emotion text with confidence
            text = f"{emotion} ({confidence:.0%})"
            cv2.putText(
                frame, text, (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2,
            )

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
