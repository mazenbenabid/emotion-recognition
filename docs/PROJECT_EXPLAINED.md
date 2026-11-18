# Project Explained — Facial Emotion Recognition

This document walks through every file in the project so you can explain
any part in an interview.

---

## 1. `train.py` — Training the CNN

### What the model looks like

The model is a **Sequential** CNN with 4 convolutional layers:

```
Conv2D(32) → Conv2D(64) → Pool → Conv2D(128) → Pool → Conv2D(256) → Pool → Dense(512) → Dense(7)
```

**Why this architecture?**
- Each Conv2D layer uses 3×3 kernels and ReLU activation
- Filter count doubles at each stage (32→64→128→256) — this is standard
  practice because deeper layers need to capture more complex features
- MaxPooling2D(2×2) halves the spatial dimensions, reducing computation
- Dropout (10% in conv blocks, 20% before output) prevents overfitting
- Final Dense(7, softmax) outputs probabilities for 7 emotion classes

### How training works

1. **Data loading:** `ImageDataGenerator` reads images from folders.
   - Training images get augmented (rotated, zoomed, flipped) to prevent
     the model from memorizing specific faces
   - Validation images are only rescaled (no augmentation) for fair evaluation

2. **Training loop:** `model.fit()` runs for 100 epochs
   - Each epoch processes ~897 batches of 32 images = ~28,700 images
   - After each epoch, the model is evaluated on the test set
   - `ModelCheckpoint` saves only the best model (highest val_accuracy)
   - `EarlyStopping` stops training if val_loss doesn't improve for 15 epochs

3. **Results:** ~62% validation accuracy
   - FER2013 is noisy — many images are ambiguous even for humans
   - Human accuracy is estimated at ~65%, so 62% is reasonable

### Key interview question: "Why not use a pretrained model?"

Good answer: "A pretrained ResNet or VGGFace would reach ~73-76% accuracy,
but the goal of this project was to understand the full CNN pipeline from
scratch — designing the architecture, tuning hyperparameters, and
understanding how convolutions learn facial features. In production,
I would absolutely use transfer learning."

---

## 2. `detect_realtime.py` — Real-Time Webcam Detection

### The inference pipeline (5 steps per frame)

```
Webcam frame → Grayscale → Face detection → Crop+Resize+Normalize → CNN predict → Draw label
```

1. **`cv2.VideoCapture(0)`** — opens the default webcam
2. **`cv2.cvtColor(frame, COLOR_BGR2GRAY)`** — Haar cascade needs grayscale
3. **`detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)`**
   - `scaleFactor=1.3` — the image is shrunk by 30% at each scale
   - `minNeighbors=3` — a face must be detected in at least 3 overlapping
     windows to be counted (reduces false positives)
4. **Preprocessing each face:**
   ```python
   resized = cv2.resize(gray_face, (48, 48))   # Match model input
   normalized = resized / 255.0                  # Scale to [0, 1]
   reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Add batch + channel dims
   ```
5. **`model.predict(input_tensor)`** — returns 7 probabilities,
   `argmax` picks the highest

### Key interview question: "Why Haar cascade instead of a deep learning face detector?"

Good answer: "Haar cascade runs at 30+ FPS on CPU with no GPU needed.
For real-time webcam applications, speed matters more than detection
accuracy. DNN-based detectors like MTCNN or RetinaFace are more accurate
(especially at angles) but add 20-50ms per frame, which would drop the
framerate below 15 FPS on a laptop CPU. If I needed better face detection
in production, I'd use MediaPipe Face Detection which is optimized for
real-time mobile/edge inference."

---

## 3. `detect_image.py` — Single Image Detection

Same pipeline as real-time, but on one image instead of a video stream.
Useful for testing and generating result screenshots.

The key addition is `--output` to save the annotated image to disk:
```bash
python detect_image.py --image happy.jpg --output results/happy_detected.jpg
```

---

## 4. Understanding the Data

### FER2013 Dataset

- **Source:** Kaggle (originally from ICML 2013 challenge)
- **Size:** ~35,887 images (28,709 train + 7,178 test)
- **Format:** 48×48 grayscale facial images
- **Classes:** 7 emotions — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Challenge:** many labels are noisy/ambiguous (a face labeled "Fear" might
  look like "Surprise" to most humans)

### Why 62% accuracy is acceptable

| Method | FER2013 Accuracy |
|--------|-----------------|
| Human performance | ~65% |
| **This project (simple CNN)** | **~62%** |
| VGG16 fine-tuned | ~72% |
| State-of-the-art (ensemble) | ~76% |

The gap between 62% and state-of-the-art comes from:
1. Deeper architectures (ResNet, VGG) with skip connections
2. Transfer learning from face recognition datasets
3. Ensemble methods (combining multiple models)
4. Better regularization (label smoothing, mixup)

---

## 5. Common Interview Questions

### Q1: "Walk me through the CNN layers."
"The input is a 48×48 grayscale image. The first Conv2D with 32 filters
extracts low-level features like edges and textures. The second layer with
64 filters combines these into patterns like eyebrow shapes or mouth curves.
The third and fourth layers (128 and 256 filters) learn higher-level features
like 'furrowed brow' or 'open mouth smile.' MaxPooling between layers reduces
spatial dimensions while keeping the strongest activations. The flatten layer
converts the 4×4×256 feature maps into a 4096-dimensional vector, and the
Dense(512) layer learns to map these features to emotion categories. The
final softmax outputs a probability distribution over 7 emotions."

### Q2: "What is data augmentation and why did you use it?"
"Data augmentation creates modified copies of training images at each epoch —
rotations, zooms, shears, and horizontal flips. This serves two purposes:
it effectively increases the training set size without collecting new data,
and it teaches the model to recognize emotions regardless of head tilt,
distance from camera, or face position. Without augmentation, the model
memorizes specific face positions and fails on new webcam feeds."

### Q3: "What's the difference between training accuracy and validation accuracy?"
"Training accuracy measures how well the model fits the training data —
it can reach 60-70% because the model has seen these images before.
Validation accuracy measures performance on unseen images and is the
real metric. When training accuracy is much higher than validation accuracy,
the model is overfitting. I used dropout and data augmentation to reduce
this gap."

### Q4: "How would you improve this project?"
"Three things: (1) Replace the simple CNN with a pretrained model like
ResNet-50 and fine-tune the last few layers — this typically reaches 72-73%
on FER2013. (2) Replace Haar cascade with MediaPipe Face Detection for
better face detection at angles and in poor lighting. (3) Add temporal
smoothing — instead of classifying each frame independently, average
predictions over 5-10 frames to reduce flickering between emotions."

### Q5: "Why did you normalize pixel values to [0,1]?"
"Raw pixels range from 0 to 255. Neural networks train better when inputs
are small because: (1) the randomly initialized weights are small, so large
inputs cause large activations and unstable gradients, (2) the Adam optimizer's
learning rate is calibrated for inputs near zero. Dividing by 255 maps
everything to [0,1], which is a simple and effective normalization."

### Q6: "What does `scaleFactor=1.3` mean in face detection?"
"Haar cascade uses a sliding window at multiple scales to find faces of
different sizes. scaleFactor=1.3 means the image is resized by 30% at each
scale level. A smaller factor (1.05) would detect faces more accurately but
take 3-4× longer. 1.3 is a good balance for real-time applications."

### Q7: "What is softmax and why use it for the output?"
"Softmax converts the 7 raw scores (logits) into probabilities that sum to 1.
If the logits are [2.0, 0.5, 0.1, 5.0, 0.3, 0.8, 0.1], softmax would output
something like [0.04, 0.01, 0.01, 0.89, 0.01, 0.01, 0.01] — the network
is 89% confident the face is Happy. This is better than just taking the max
because it gives us a confidence score. If the highest probability is only
30%, we know the prediction is uncertain."

### Q8: "Why categorical cross-entropy loss?"
"This is a 7-class classification problem. Categorical cross-entropy measures
how far the predicted probability distribution is from the true label. If the
true label is Happy (index 3), the loss is -log(predicted_probability[3]).
So if the model predicts 90% Happy, the loss is -log(0.9) = 0.105 (low).
If it predicts 10% Happy, the loss is -log(0.1) = 2.302 (high). This
directly penalizes confident wrong predictions."

### Q9: "What is the role of dropout in your model?"
"Dropout randomly sets a fraction of neuron outputs to zero during training.
In my model, 10% dropout after each pooling layer and 20% before the final
layer. This forces the network to learn redundant representations — no
single neuron can memorize a specific pattern. It's like training an ensemble
of smaller networks. During inference, dropout is disabled and all neurons
contribute, which averages these sub-networks."

### Q10: "Could this run on a Raspberry Pi or mobile phone?"
"The model has only 2.5M parameters and the inference takes <5ms on a laptop
CPU, so yes — it could run on a Raspberry Pi 4 at ~15 FPS. For mobile,
I would convert the model to TensorFlow Lite (quantized to INT8) which would
reduce the model size from ~10MB to ~2.5MB and speed up inference. The Haar
cascade is also very lightweight compared to deep learning face detectors."
