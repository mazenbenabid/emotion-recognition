"""
Detect and classify emotions on a single image.

Usage:
    python detect_image.py --image path/to/image.jpg
    python detect_image.py --image path/to/image.jpg --model model_file.keras
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

BOX_COLOR = (50, 50, 255)
LABEL_BG_COLOR = (50, 50, 255)
TEXT_COLOR = (255, 255, 255)


def preprocess_face(gray_face: np.ndarray) -> np.ndarray:
    """Resize to 48×48, normalize to [0,1], reshape for batch prediction."""
    resized = cv2.resize(gray_face, (48, 48))
    normalized = resized / 255.0
    return np.reshape(normalized, (1, 48, 48, 1))


def main():
    parser = argparse.ArgumentParser(description="Detect emotions in an image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model", type=str, default="model_file.keras",
                        help="Path to the trained Keras model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the annotated image (optional)")
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Load Haar-cascade face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load Haar-cascade classifier.")
        return

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"Error: Could not load image '{args.image}'.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    print(f"Detected {len(faces)} face(s):\n")

    for i, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y : y + h, x : x + w]
        input_tensor = preprocess_face(face_roi)

        predictions = model.predict(input_tensor, verbose=0)
        label_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][label_idx]
        emotion = EMOTION_LABELS[label_idx]

        print(f"  Face {i + 1}: {emotion} ({confidence:.1%})")

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), LABEL_BG_COLOR, -1)
        cv2.putText(
            frame, f"{emotion} ({confidence:.0%})", (x + 5, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2,
        )

    # Save or display
    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"\nAnnotated image saved to {args.output}")
    else:
        cv2.imshow("Emotion Detection", frame)
        print("\nPress any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
