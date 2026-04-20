import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────
def generate_dataset(n_samples=800):
    images = []
    labels = []

    per_class = n_samples // 10

    for digit in range(10):
        for _ in range(per_class):
            img = np.zeros((1, 10))

            if digit > 0:
                pos = np.random.choice(10, digit, replace=False)
                img[0, pos] = 1

            images.append(img)
            labels.append(digit)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# ─────────────────────────────────────────────
# 2. TRAIN TEST SPLIT
# ─────────────────────────────────────────────
def split_data(images, labels):
    return train_test_split(images, labels, test_size=0.2) # can use stratifiy = labels..


# ─────────────────────────────────────────────
# 3. BUILD CNN MODEL (IN-BUILT)
# ─────────────────────────────────────────────
def build_model():

    model = Sequential([

        # Convolution
        Conv2D(8, (1,3), activation='relu', input_shape=(1,10,1)),

        # Pooling
        MaxPooling2D(pool_size=(1,2)),

        # Flatten
        Flatten(),

        # Fully connected
        Dense(64, activation='relu'),

        # Output layer
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, epochs=20):
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    return model


# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    return acc


# ─────────────────────────────────────────────
# 6. PREDICT
# ─────────────────────────────────────────────
def predict_digit(model, image):
    image = image.reshape(1,1,10,1)   # VERY IMPORTANT
    probs = model.predict(image)
    return np.argmax(probs)
































# """
# cnn_model.py  (fixed version)
# ==============================
# Custom CNN for 1×10 binary digit images — pure NumPy.
# Fixes:
#   1. Dataset noise so the task is non-trivial
#   2. He weight initialisation to prevent saturation
#   3. Gradient clipping
#   4. Proper stratified split
# """

# import numpy as np
# from collections import Counter


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. DATASET
# # ─────────────────────────────────────────────────────────────────────────────

# def generate_dataset(n_samples: int = 800, noise_prob: float = 0.08, seed: int = 42):
#     """
#     1×10 binary images where digit k has exactly k black cells.
#     noise_prob: probability of flipping any individual cell (adds realism).
#     Returns images (n, 1, 10) and labels (n,).
#     """
#     rng = np.random.default_rng(seed)
#     images, labels = [], []

#     per_class = n_samples // 10
#     remainder = n_samples - per_class * 10

#     for digit in range(10):
#         count = per_class + (1 if digit < remainder else 0)
#         for _ in range(count):
#             img = np.zeros((1, 10), dtype=np.float32)
#             if digit > 0:
#                 pos = rng.choice(10, size=digit, replace=False)
#                 img[0, pos] = 1.0
#             # Add noise: flip cells randomly
#             if noise_prob > 0:
#                 mask = rng.random((1, 10)) < noise_prob
#                 img  = np.abs(img - mask.astype(np.float32))
#                 img  = np.clip(img, 0, 1)
#             images.append(img)
#             labels.append(digit)

#     images = np.array(images)
#     labels = np.array(labels)
#     idx    = rng.permutation(len(labels))
#     return images[idx], labels[idx]


# def train_test_split(images, labels, test_ratio=0.2, seed=42):
#     """Stratified 80-20 split."""
#     rng = np.random.default_rng(seed)
#     train_idx, test_idx = [], []
#     for cls in range(10):
#         cls_idx = np.where(labels == cls)[0]
#         rng.shuffle(cls_idx)
#         n_test = max(1, int(len(cls_idx) * test_ratio))
#         test_idx.extend(cls_idx[:n_test])
#         train_idx.extend(cls_idx[n_test:])
#     return (images[train_idx], labels[train_idx],
#             images[test_idx],  labels[test_idx])


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. CUSTOM LAYER FUNCTIONS
# # ─────────────────────────────────────────────────────────────────────────────

# def pad(image: np.ndarray, pad_width: int, value: float = 0.0) -> np.ndarray:
#     """Zero-pad a 2-D (H, W) array on all sides."""
#     if pad_width == 0:
#         return image
#     H, W = image.shape
#     out  = np.full((H + 2*pad_width, W + 2*pad_width), value, dtype=image.dtype)
#     out[pad_width:pad_width+H, pad_width:pad_width+W] = image
#     return out


# def convolve(image: np.ndarray, kernel: np.ndarray,
#              stride: int = 1, padding: int = 0) -> np.ndarray:
#     """2-D cross-correlation (what ML calls convolution)."""
#     image  = pad(image, padding)
#     H, W   = image.shape
#     kH, kW = kernel.shape
#     out_H  = (H - kH) // stride + 1
#     out_W  = (W - kW) // stride + 1
#     out    = np.zeros((out_H, out_W), dtype=np.float32)
#     for i in range(out_H):
#         for j in range(out_W):
#             out[i, j] = np.sum(
#                 image[i*stride:i*stride+kH, j*stride:j*stride+kW] * kernel)
#     return out


# def relu(x: np.ndarray) -> np.ndarray:
#     return np.maximum(0.0, x)


# def max_pool(fm: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
#     """2-D max pooling."""
#     H, W  = fm.shape
#     out_H = max(1, (H - pool_size) // stride + 1)
#     out_W = max(1, (W - pool_size) // stride + 1)
#     out   = np.zeros((out_H, out_W), dtype=np.float32)
#     for i in range(out_H):
#         for j in range(out_W):
#             out[i, j] = np.max(
#                 fm[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
#     return out


# def avg_pool(fm: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
#     """2-D average pooling."""
#     H, W  = fm.shape
#     out_H = max(1, (H - pool_size) // stride + 1)
#     out_W = max(1, (W - pool_size) // stride + 1)
#     out   = np.zeros((out_H, out_W), dtype=np.float32)
#     for i in range(out_H):
#         for j in range(out_W):
#             out[i, j] = np.mean(
#                 fm[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size])
#     return out


# def fully_connected(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
#     """Linear layer: W @ x + b"""
#     return W @ x + b


# def softmax(x: np.ndarray) -> np.ndarray:
#     """Numerically stable softmax."""
#     e = np.exp(x - np.max(x))
#     return e / (e.sum() + 1e-9)


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. CNN CLASS
# # ─────────────────────────────────────────────────────────────────────────────

# class SimpleCNN:
#     """
#     Architecture:
#       Input  (1, 10)
#       Conv   n_filters × (1, kernel_size), stride=1, no padding
#       ReLU
#       MaxPool (1, pool_size), stride=pool_stride
#       Flatten
#       FC1    → 64  (ReLU)
#       FC2    → 10  (Softmax)

#     Training: SGD + cross-entropy + gradient clipping.
#     """

#     def __init__(self, kernel_size=3, pool_size=2, pool_stride=1,
#                  n_filters=8, seed=0):
#         self.kernel_size = kernel_size
#         self.pool_size   = pool_size
#         self.pool_stride = pool_stride
#         self.n_filters   = n_filters

#         rng    = np.random.default_rng(seed)
#         fan_in = kernel_size

#         # He initialisation for conv kernels
#         self.kernels   = (rng.standard_normal((n_filters, 1, kernel_size))
#                           * np.sqrt(2.0 / fan_in)).astype(np.float32)
#         self.conv_bias = np.zeros(n_filters, dtype=np.float32)
#         self._fc_initialised = False

#     # ── internal helpers ───────────────────────────────────────────────────

#     def _conv_forward(self, image: np.ndarray) -> np.ndarray:
#         maps = []
#         for f in range(self.n_filters):
#             fm = convolve(image, self.kernels[f], stride=1, padding=0)
#             fm = relu(fm + self.conv_bias[f])
#             fm = max_pool(fm, pool_size=self.pool_size, stride=self.pool_stride)
#             maps.append(fm)
#         return np.array(maps)      # (n_filters, 1, out_W)

#     def _init_fc(self, flat_dim: int):
#         rng    = np.random.default_rng(42)
#         hidden = 64
#         self.W1 = (rng.standard_normal((hidden, flat_dim))
#                    * np.sqrt(2.0 / flat_dim)).astype(np.float32)
#         self.b1 = np.zeros(hidden, dtype=np.float32)
#         self.W2 = (rng.standard_normal((10, hidden))
#                    * np.sqrt(2.0 / hidden)).astype(np.float32)
#         self.b2 = np.zeros(10, dtype=np.float32)
#         self._fc_initialised = True

#     # ── forward pass ───────────────────────────────────────────────────────

#     def forward(self, image: np.ndarray):
#         feat = self._conv_forward(image)
#         flat = feat.flatten()

#         if not self._fc_initialised:
#             self._init_fc(len(flat))

#         h1     = relu(fully_connected(flat, self.W1, self.b1))
#         logits = fully_connected(h1, self.W2, self.b2)
#         probs  = softmax(logits)

#         cache = dict(image=image, feat=feat, flat=flat,
#                      h1=h1, logits=logits, probs=probs)
#         return probs, cache

#     # ── backward pass ──────────────────────────────────────────────────────

#     def _backward(self, cache, label: int, lr: float, clip: float = 5.0):
#         probs = cache['probs']
#         h1    = cache['h1']
#         flat  = cache['flat']

#         # Softmax + cross-entropy gradient
#         dlogits        = probs.copy()
#         dlogits[label] -= 1.0

#         dW2 = np.outer(dlogits, h1)
#         db2 = dlogits.copy()

#         dh1          = self.W2.T @ dlogits
#         dh1[h1 <= 0] = 0.0           # ReLU mask

#         dW1 = np.outer(dh1, flat)
#         db1 = dh1.copy()

#         # Gradient clipping
#         for g in [dW2, db2, dW1, db1]:
#             np.clip(g, -clip, clip, out=g)

#         self.W2 -= lr * dW2
#         self.b2 -= lr * db2
#         self.W1 -= lr * dW1
#         self.b1 -= lr * db1

#     # ── training loop ──────────────────────────────────────────────────────

#     def train(self, images, labels, epochs=30, lr=0.01, verbose=True):
#         n   = len(labels)
#         rng = np.random.default_rng(99)
#         losses = []

#         for ep in range(1, epochs + 1):
#             idx   = rng.permutation(n)
#             total = 0.0
#             for i in idx:
#                 probs, cache = self.forward(images[i])
#                 total += -np.log(probs[labels[i]] + 1e-9)
#                 self._backward(cache, labels[i], lr)
#             avg = total / n
#             losses.append(avg)
#             if verbose and (ep % 5 == 0 or ep == 1):
#                 print(f"  Epoch {ep:3d}/{epochs}  loss={avg:.4f}")
#         return losses

#     # ── inference ──────────────────────────────────────────────────────────

#     def predict(self, image: np.ndarray) -> int:
#         probs, _ = self.forward(image)
#         return int(np.argmax(probs))

#     def predict_proba(self, image: np.ndarray) -> np.ndarray:
#         probs, _ = self.forward(image)
#         return probs

#     def evaluate(self, images, labels) -> float:
#         correct = sum(self.predict(images[i]) == labels[i]
#                       for i in range(len(labels)))
#         return correct / len(labels)


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. QUICK SANITY CHECK
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=== Generating dataset (with noise) ===")
#     images, labels = generate_dataset(n_samples=800, noise_prob=0.08)
#     print(f"Class distribution: {Counter(labels.tolist())}")

#     X_train, y_train, X_test, y_test = train_test_split(images, labels)
#     print(f"Train: {len(y_train)}  Test: {len(y_test)}")

#     print("\n=== Training ===")
#     model = SimpleCNN(kernel_size=3, pool_size=2, pool_stride=1, n_filters=8)
#     model.train(X_train, y_train, epochs=40, lr=0.01)

#     print(f"\nTrain acc : {model.evaluate(X_train, y_train)*100:.1f}%")
#     print(f"Test  acc : {model.evaluate(X_test,  y_test)*100:.1f}%")

#     print("\nPer-digit test accuracy:")
#     for d in range(10):
#         idx = np.where(y_test == d)[0]
#         acc = sum(model.predict(X_test[i]) == d for i in idx) / len(idx)
#         print(f"  Digit {d}: {acc*100:.0f}%  (n={len(idx)})")
