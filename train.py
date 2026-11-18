"""
Train a CNN for facial emotion recognition on the FER2013 dataset.

Architecture: 4 convolutional blocks (32→64→128→256 filters) with max-pooling
and dropout, followed by a 512-unit dense layer and 7-class softmax output.

Dataset: FER2013 — 48×48 grayscale facial images in 7 emotion categories:
    Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

Usage:
    python train.py --data_dir archive --epochs 100 --batch_size 32
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ── Emotion labels (FER2013 class order) ────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = (48, 48)


def build_model() -> Sequential:
    """
    Build the CNN architecture.

    4 convolutional blocks with increasing filter depth:
        Block 1: Conv2D(32) → ReLU
        Block 2: Conv2D(64) → ReLU → MaxPool(2×2) → Dropout(0.1)
        Block 3: Conv2D(128) → ReLU → MaxPool(2×2) → Dropout(0.1)
        Block 4: Conv2D(256) → ReLU → MaxPool(2×2) → Dropout(0.1)

    Classifier head: Flatten → Dense(512) → Dropout(0.2) → Dense(7, softmax)
    """
    model = Sequential([
        # Block 1: 48×48×1 → 46×46×32
        Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),

        # Block 2: 46×46×32 → 44×44×64 → 22×22×64
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        # Block 3: 22×22×64 → 20×20×128 → 10×10×128
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        # Block 4: 10×10×128 → 8×8×256 → 4×4×256
        Conv2D(256, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        # Classifier head: 4×4×256 = 4096 → 512 → 7
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.2),
        Dense(len(EMOTION_LABELS), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_images(directory: str) -> int:
    """Count total image files in a directory tree."""
    total = 0
    for _, _, files in os.walk(directory):
        total += len(files)
    return total


def create_generators(data_dir: str, batch_size: int):
    """
    Create training and validation data generators.

    Training data is augmented with rotation, shear, zoom, and horizontal flip
    to reduce overfitting and improve generalization.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        color_mode="grayscale",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        color_mode="grayscale",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    return train_gen, val_gen


def plot_training_history(history, output_path: str = "results/training_history.png"):
    """Plot and save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(history.history["loss"], label="Train", linewidth=2)
    ax2.plot(history.history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition CNN")
    parser.add_argument("--data_dir", type=str, default="archive",
                        help="Root directory containing train/ and test/ folders")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--output", type=str, default="model_file.keras",
                        help="Path to save the trained model")
    args = parser.parse_args()

    # Count dataset images
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    num_train = count_images(train_dir)
    num_test = count_images(test_dir)
    print(f"Training images:   {num_train}")
    print(f"Validation images: {num_test}")
    print(f"Emotion classes:   {EMOTION_LABELS}")
    print()

    # Build model
    model = build_model()
    model.summary()
    print()

    # Data generators
    train_gen, val_gen = create_generators(args.data_dir, args.batch_size)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            args.output,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Train
    history = model.fit(
        train_gen,
        steps_per_epoch=num_train // args.batch_size,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=num_test // args.batch_size,
        callbacks=callbacks,
    )

    # Save final model (best weights already saved by ModelCheckpoint)
    model.save(args.output)
    print(f"\nModel saved to {args.output}")

    # Plot training history
    plot_training_history(history)

    # Print best results
    best_epoch = np.argmax(history.history["val_accuracy"])
    best_val_acc = history.history["val_accuracy"][best_epoch]
    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch + 1}")


if __name__ == "__main__":
    main()
