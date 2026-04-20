"""
cnn_model.py  — FULL EXPLANATION VERSION
----------------------------------------

GOAL:
-----
Build a CNN from scratch (NumPy only) to classify digits (0–9)
based on 1×10 binary images.

PIPELINE:
---------
Dataset → Split → Conv → ReLU → Pool → Flatten → FC → Softmax → Prediction → Training
"""

import numpy as np
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n_samples=800, noise_prob=0.08, seed=42):
    """
    PURPOSE:
    --------
    Create dataset of 1×10 binary images.

    LOGIC:
    ------
    Digit k → exactly k cells = 1 (black), rest = 0.

    EXAMPLE:
    --------
    digit = 3 → [0 1 0 1 0 0 1 0 0 0]

    ADDITION:
    ---------
    Noise flips some cells randomly → makes problem harder.
    """

    rng = np.random.default_rng(seed)
    images, labels = [], []

    per_class = n_samples // 10

    for digit in range(10):  # digits 0–9

        for _ in range(per_class):

            # create empty image
            img = np.zeros((1, 10), dtype=np.float32)

            # randomly place 'digit' number of 1s
            if digit > 0:
                pos = rng.choice(10, size=digit, replace=False)
                img[0, pos] = 1.0

            # add noise (flip cells randomly)
            if noise_prob > 0:
                mask = rng.random((1, 10)) < noise_prob
                img  = np.abs(img - mask.astype(np.float32))

            images.append(img)
            labels.append(digit)

    # convert to arrays
    images = np.array(images)
    labels = np.array(labels)

    # shuffle dataset
    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


def train_test_split(images, labels, test_ratio=0.2):
    """
    PURPOSE:
    --------
    Split data into train & test sets.

    IMPORTANT:
    ----------
    Stratified split → each digit appears equally in both sets.
    """

    train_idx, test_idx = [], []

    for cls in range(10):

        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        n_test = int(len(cls_idx) * test_ratio)

        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    return (
        images[train_idx], labels[train_idx],
        images[test_idx],  labels[test_idx]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CNN BASIC OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def pad(image, pad_width):
    """
    PURPOSE:
    --------
    Add border around image.

    EXAMPLE:
    --------
    [1 2] → pad=1 →

    [0 0 0 0]
    [0 1 2 0]
    [0 0 0 0]
    """

    if pad_width == 0:
        return image

    H, W = image.shape
    out = np.zeros((H + 2*pad_width, W + 2*pad_width))

    out[pad_width:pad_width+H, pad_width:pad_width+W] = image
    return out


def convolve(image, kernel, stride=1, padding=0):
    """
    PURPOSE:
    --------
    Apply kernel → extract features.

    CORE IDEA:
    ----------
    Slide kernel → multiply → sum → output value
    """

    image = pad(image, padding)

    H, W = image.shape
    kH, kW = kernel.shape

    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1

    out = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):

            # extract window
            window = image[i:i+kH, j:j+kW]

            # multiply + sum
            out[i, j] = np.sum(window * kernel)

    return out


def relu(x):
    """ReLU activation → remove negatives"""
    return np.maximum(0, x)


def max_pool(fm, pool_size=2, stride=1):
    """
    PURPOSE:
    --------
    Reduce size while keeping strongest values.
    """

    H, W = fm.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1

    out = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            out[i, j] = np.max(
                fm[i:i+pool_size, j:j+pool_size]
            )

    return out


def fully_connected(x, W, b):
    """Linear layer"""
    return W @ x + b


def softmax(x):
    """Convert scores → probabilities"""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

# class SimpleCNN:

#     def __init__(self, kernel_size=3, n_filters=8):

#         self.kernel_size = kernel_size
#         self.n_filters = n_filters

#         # initialize kernels
#         self.kernels = np.random.randn(n_filters, 1, kernel_size)

#         self._fc_initialised = False

#     def _conv_forward(self, image):
#         """
#         Apply all filters → get feature maps
#         """

#         maps = []

#         for f in range(self.n_filters):
#             fm = convolve(image, self.kernels[f])
#             fm = relu(fm)
#             fm = max_pool(fm)

#             maps.append(fm)

#         return np.array(maps)


#     def _init_fc(self, flat_dim):
#         """Initialize fully connected layers"""

#         self.W1 = np.random.randn(64, flat_dim)
#         self.b1 = np.zeros(64)

#         self.W2 = np.random.randn(10, 64)
#         self.b2 = np.zeros(10)

#         self._fc_initialised = True


#     def forward(self, image):
#         """
#         FULL FORWARD PASS:

#         Conv → ReLU → Pool → Flatten → FC → ReLU → FC → Softmax
#         """

#         feat = self._conv_forward(image)

#         flat = feat.flatten()

#         if not self._fc_initialised:
#             self._init_fc(len(flat))

#         h1 = relu(fully_connected(flat, self.W1, self.b1))

#         logits = fully_connected(h1, self.W2, self.b2)

#         probs = softmax(logits)

#         cache = (flat, h1, probs)

#         return probs, cache


#     def _backward(self, cache, label, lr):
#         """
#         PURPOSE:
#         --------
#         Update weights using gradient descent
#         """

#         flat, h1, probs = cache

#         # compute error
#         dlogits = probs.copy()
#         dlogits[label] -= 1

#         # gradients FC2
#         dW2 = np.outer(dlogits, h1)
#         db2 = dlogits

#         # backprop to hidden
#         dh1 = self.W2.T @ dlogits
#         dh1[h1 <= 0] = 0

#         # gradients FC1
#         dW1 = np.outer(dh1, flat)
#         db1 = dh1

#         # update weights
#         self.W2 -= lr * dW2
#         self.b2 -= lr * db2
#         self.W1 -= lr * dW1
#         self.b1 -= lr * db1


#     def train(self, images, labels, epochs=30, lr=0.01):
#         """
#         TRAINING LOOP:
#         --------------
#         forward → loss → backward → update
#         """

#         for ep in range(epochs):
#             for i in range(len(labels)):
#                 probs, cache = self.forward(images[i])
#                 self._backward(cache, labels[i], lr)


#     def predict(self, image):
#         """Return predicted digit"""
#         probs, _ = self.forward(image)
#         return np.argmax(probs)


#     def evaluate(self, images, labels):
#         """Compute accuracy"""
#         correct = 0
#         for i in range(len(labels)):
#             if self.predict(images[i]) == labels[i]:
#                 correct += 1
#         return correct / len(labels)


class SimpleCNN:
    """
    PURPOSE:
    --------
    Build a CNN model to classify digits (0–9) from 1×10 binary images.

    PIPELINE:
    ---------
    Input → Conv → ReLU → Pool → Flatten → FC → ReLU → FC → Softmax

    IDEA:
    -----
    - Convolution → extract features (patterns of 1s)
    - Pooling → reduce size
    - Fully Connected → classify digit
    """

    def __init__(self, kernel_size=3, pool_size=2, pool_stride=1,
                 n_filters=8, seed=0):

        """
        PURPOSE:
        --------
        Initialize model parameters (filters, biases, etc.)

        LOGIC:
        ------
        - Define kernel size, pooling size, number of filters
        - Initialize convolution kernels (random)
        - FC layers are initialized later (depends on input size)
        """

        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.pool_stride = pool_stride
        self.n_filters   = n_filters

        rng = np.random.default_rng(seed)

        # He initialization → keeps values stable during training
        fan_in = kernel_size

        # shape = (number of filters, height=1, width=kernel_size)
        self.kernels = (
            rng.standard_normal((n_filters, 1, kernel_size))
            * np.sqrt(2.0 / fan_in)
        ).astype(np.float32)

        # one bias per filter
        self.conv_bias = np.zeros(n_filters, dtype=np.float32)

        # FC layers will be initialized after seeing first input
        self._fc_initialised = False


    # ─────────────────────────────────────────────────────────────
    # CONVOLUTION FORWARD
    # ─────────────────────────────────────────────────────────────

    def _conv_forward(self, image: np.ndarray) -> np.ndarray:
        """
        PURPOSE:
        --------
        Apply all filters → extract feature maps

        LOGIC:
        ------
        For each filter:
            Convolution → ReLU → Pool → store output

        OUTPUT:
        -------
        shape = (n_filters, 1, output_width)
        """

        maps = []

        for f in range(self.n_filters):

            # STEP 1: Convolution → extract pattern
            fm = convolve(image, self.kernels[f], stride=1, padding=0)

            # STEP 2: Add bias + ReLU → remove negative values
            fm = relu(fm + self.conv_bias[f])

            # STEP 3: Pooling → reduce size, keep important values
            fm = max_pool(fm, pool_size=self.pool_size, stride=self.pool_stride)

            # store feature map
            maps.append(fm)

        return np.array(maps)


    # ─────────────────────────────────────────────────────────────
    # INITIALIZE FULLY CONNECTED LAYERS
    # ─────────────────────────────────────────────────────────────

    def _init_fc(self, flat_dim: int):
        """
        PURPOSE:
        --------
        Initialize weights for fully connected layers

        LOGIC:
        ------
        - Input size = flattened size
        - Hidden layer = 64 neurons
        - Output layer = 10 neurons (digits 0–9)
        """

        rng = np.random.default_rng(42)
        hidden = 64

        # FC1 weights
        self.W1 = (rng.standard_normal((hidden, flat_dim))
                   * np.sqrt(2.0 / flat_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)

        # FC2 weights
        self.W2 = (rng.standard_normal((10, hidden))
                   * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(10, dtype=np.float32)

        self._fc_initialised = True


    # ─────────────────────────────────────────────────────────────
    # FORWARD PASS (PREDICTION)
    # ─────────────────────────────────────────────────────────────

    def forward(self, image: np.ndarray):
        """
        PURPOSE:
        --------
        Perform full forward pass → get prediction

        STEPS:
        ------
        1. Convolution → feature maps
        2. Flatten → convert to vector
        3. FC1 + ReLU
        4. FC2
        5. Softmax → probabilities
        """

        # STEP 1: Feature extraction
        feat = self._conv_forward(image)

        # STEP 2: Flatten (3D → 1D)
        flat = feat.flatten()

        # initialize FC if first time
        if not self._fc_initialised:
            self._init_fc(len(flat))

        # STEP 3: First FC layer + ReLU
        h1 = relu(fully_connected(flat, self.W1, self.b1))

        # STEP 4: Second FC layer (logits)
        logits = fully_connected(h1, self.W2, self.b2)

        # STEP 5: Softmax → probabilities
        probs = softmax(logits)

        # store values for backpropagation
        cache = dict(image=image, feat=feat, flat=flat,
                     h1=h1, logits=logits, probs=probs)

        return probs, cache


    # ─────────────────────────────────────────────────────────────
    # BACKWARD PASS (LEARNING)
    # ─────────────────────────────────────────────────────────────

    def _backward(self, cache, label: int, lr: float, clip: float = 5.0):
        """
        PURPOSE:
        --------
        Update weights using gradient descent

        LOGIC:
        ------
        error = predicted - actual
        → compute gradients
        → update weights
        """

        probs = cache['probs']
        h1    = cache['h1']
        flat  = cache['flat']

        # STEP 1: compute error (softmax + cross entropy)
        dlogits = probs.copy()
        dlogits[label] -= 1.0

        # STEP 2: gradients for FC2
        dW2 = np.outer(dlogits, h1)
        db2 = dlogits.copy()

        # STEP 3: backprop to hidden layer
        dh1 = self.W2.T @ dlogits

        # apply ReLU derivative
        dh1[h1 <= 0] = 0.0

        # STEP 4: gradients for FC1
        dW1 = np.outer(dh1, flat)
        db1 = dh1.copy()

        # STEP 5: gradient clipping (avoid explosion)
        for g in [dW2, db2, dW1, db1]:
            np.clip(g, -clip, clip, out=g)

        # STEP 6: update weights
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


    # ─────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────

    def train(self, images, labels, epochs=30, lr=0.01):
        """
        PURPOSE:
        --------
        Train model on dataset

        PROCESS:
        --------
        For each epoch:
            shuffle data
            forward → compute loss → backward
        """

        n = len(labels)

        for ep in range(epochs):

            idx = np.random.permutation(n)

            for i in idx:
                probs, cache = self.forward(images[i])

                # loss (cross entropy)
                loss = -np.log(probs[labels[i]] + 1e-9)

                # update weights
                self._backward(cache, labels[i], lr)


    # ─────────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> int:
        """
        PURPOSE:
        --------
        Return predicted digit
        """
        probs, _ = self.forward(image)
        return int(np.argmax(probs))


    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """
        PURPOSE:
        --------
        Return probability for each digit
        """
        probs, _ = self.forward(image)
        return probs


    def evaluate(self, images, labels) -> float:
        """
        PURPOSE:
        --------
        Compute accuracy of model
        """
        correct = sum(self.predict(images[i]) == labels[i]
                      for i in range(len(labels)))
        return correct / len(labels)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION (TEST)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # generate dataset
    images, labels = generate_dataset()

    # split data
    X_train, y_train, X_test, y_test = train_test_split(images, labels)

    # create model
    model = SimpleCNN()

    # train model
    model.train(X_train, y_train)

    # evaluate
    print("Train Accuracy:", model.evaluate(X_train, y_train))
    print("Test Accuracy :", model.evaluate(X_test, y_test))
    ##########################################################################################

"""
app.py — STREAMLIT FRONTEND

PURPOSE:
--------
This file creates a web app to:
1. Train CNN model
2. Predict digit from input
3. Visualize results

FLOW:
-----
User → Set parameters → Train model → Give input → Get prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

# import our CNN code
from cnn_model import SimpleCNN, generate_dataset, train_test_split


# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────

"""
PURPOSE:
--------
Configure Streamlit page UI
"""

st.set_page_config(
    page_title="CNN Digit Predictor",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 CNN Digit Predictor")

st.markdown(
    """
    This app trains a CNN and predicts digits from 1×10 images.
    """
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR INPUT (HYPERPARAMETERS)
# ─────────────────────────────────────────────────────────────

"""
PURPOSE:
--------
Allow user to change model parameters
"""

st.sidebar.header("⚙️ Hyper-parameters")

kernel_size = st.sidebar.slider("Kernel size", 1, 5, 3)
pool_size   = st.sidebar.slider("Pool size", 1, 4, 2)
pool_stride = st.sidebar.slider("Pool stride", 1, 3, 1)
n_filters   = st.sidebar.slider("Filters", 2, 16, 8)

epochs      = st.sidebar.slider("Epochs", 5, 80, 40)
lr          = st.sidebar.select_slider(
    "Learning rate",
    options=[0.001, 0.005, 0.01, 0.05],
    value=0.01
)

n_samples   = st.sidebar.slider("Dataset size", 200, 1200, 800)
noise_prob  = st.sidebar.slider("Noise", 0.0, 0.2, 0.08)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def render_strip(image_1x10):
    """
    PURPOSE:
    --------
    Convert 1×10 array → visual image (black/white boxes)
    """

    fig, ax = plt.subplots(figsize=(6, 1))

    row = image_1x10[0]

    for j, val in enumerate(row):

        # black if value=1, else white
        color = "black" if val > 0.5 else "white"

        rect = mpatches.Rectangle((j, 0), 1, 1, facecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    return fig


def make_clean_sample(digit):
    """
    PURPOSE:
    --------
    Generate perfect (no noise) image for a digit

    EXAMPLE:
    --------
    digit=3 → exactly 3 ones
    """

    img = np.zeros((1, 10))

    if digit > 0:
        pos = np.random.choice(10, digit, replace=False)
        img[0, pos] = 1

    return img


def parse_uploaded_image(uploaded_file):
    """
    PURPOSE:
    --------
    Convert uploaded image → 1×10 format for CNN

    STEPS:
    ------
    1. Convert to grayscale
    2. Resize to (10,1)
    3. Normalize
    4. Convert to binary
    """

    img = Image.open(uploaded_file).convert("L")

    img = img.resize((10, 1))

    arr = np.array(img) / 255.0

    # threshold → binary
    arr = (arr < 0.5).astype(np.float32)

    return arr


# ─────────────────────────────────────────────────────────────
# SESSION STORAGE
# ─────────────────────────────────────────────────────────────

"""
PURPOSE:
--------
Store model + results so they persist across UI interactions
"""

if "model" not in st.session_state:
    st.session_state.model = None


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────

tab_train, tab_predict = st.tabs(["Train", "Predict"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — TRAIN MODEL
# ─────────────────────────────────────────────────────────────

with tab_train:

    st.subheader("Train CNN")

    if st.button("Train Model"):

        # STEP 1: Generate dataset
        images, labels = generate_dataset(n_samples, noise_prob)

        # STEP 2: Split dataset
        X_train, y_train, X_test, y_test = train_test_split(images, labels)

        # STEP 3: Create model
        model = SimpleCNN(
            kernel_size=kernel_size,
            pool_size=pool_size,
            pool_stride=pool_stride,
            n_filters=n_filters
        )

        # STEP 4: Train model
        model.train(X_train, y_train, epochs=epochs, lr=lr)

        # store model
        st.session_state.model = model

        # evaluate
        train_acc = model.evaluate(X_train, y_train)
        test_acc  = model.evaluate(X_test, y_test)

        st.success("Training Done!")

        st.write("Train Accuracy:", train_acc)
        st.write("Test Accuracy:", test_acc)


# ─────────────────────────────────────────────────────────────
# TAB 2 — PREDICT
# ─────────────────────────────────────────────────────────────

with tab_predict:

    st.subheader("Predict Digit")

    if st.session_state.model is None:
        st.warning("Train model first!")
    else:

        model = st.session_state.model

        option = st.radio("Choose input", ["Generate", "Upload"])

        image = None

        # OPTION 1: Generate
        if option == "Generate":

            digit = st.selectbox("Digit", list(range(10)))

            if st.button("Predict"):
                image = make_clean_sample(digit)

        # OPTION 2: Upload
        else:
            uploaded = st.file_uploader("Upload image")

            if uploaded:
                image = parse_uploaded_image(uploaded)

        # RUN PREDICTION
        if image is not None:

            st.write("Input image:")

            fig = render_strip(image)
            st.pyplot(fig)

            probs = model.predict_proba(image)

            pred = np.argmax(probs)

            st.write("Predicted Digit:", pred)
            st.write("Confidence:", probs[pred])

## import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from PIL import Image

from cnn_model import SimpleCNN, generate_dataset, train_test_split


# ─────────────────────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CNN Digit Predictor", page_icon="🔢", layout="wide")

st.title("🔢 CNN Digit Predictor")

st.markdown("Train a CNN and predict digits from 1×10 binary images.")


# ─────────────────────────────────────────────────────────────
# SIDEBAR (FIXED PARAMETERS)
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Hyperparameters")

kernel_size = st.sidebar.slider("Kernel size", 1, 5, 3)
pool_size   = st.sidebar.slider("Pool size", 1, 4, 2)

# ✅ FIX: force stride = pool_size
pool_stride = pool_size

n_filters   = st.sidebar.slider("Filters", 4, 32, 16)
epochs      = st.sidebar.slider("Epochs", 5, 60, 40)
lr          = st.sidebar.select_slider(
    "Learning rate",
    options=[0.001, 0.005, 0.01],
    value=0.01
)

n_samples   = st.sidebar.slider("Dataset size", 200, 1200, 800)
noise_prob  = st.sidebar.slider("Noise", 0.0, 0.1, 0.02)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def render_strip(image):
    fig, ax = plt.subplots(figsize=(6, 1))
    row = image[0]

    for j, val in enumerate(row):
        color = "black" if val > 0.5 else "white"
        rect = mpatches.Rectangle((j, 0), 1, 1, facecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")

    return fig


def make_clean_sample(digit):
    img = np.zeros((1, 10))
    if digit > 0:
        pos = np.random.choice(10, digit, replace=False)
        img[0, pos] = 1
    return img


def parse_uploaded_image(uploaded_file, invert=False):
    """
    FIXED:
    - Better thresholding
    - Optional inversion
    """

    img = Image.open(uploaded_file).convert("L")

    # resize to 1×10
    img = img.resize((10, 1))

    arr = np.array(img) / 255.0

    # adaptive threshold
    threshold = np.mean(arr)
    arr = (arr < threshold).astype(np.float32)

    # optional inversion
    if invert:
        arr = 1 - arr

    return arr


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = None


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_train, tab_predict = st.tabs(["Train", "Predict"])


# ─────────────────────────────────────────────────────────────
# TRAIN TAB
# ─────────────────────────────────────────────────────────────
with tab_train:

    st.subheader("Train Model")

    if st.button("Train Model"):

        # generate dataset
        images, labels = generate_dataset(n_samples, noise_prob)

        X_train, y_train, X_test, y_test = train_test_split(images, labels)

        # create model
        model = SimpleCNN(
            kernel_size=kernel_size,
            pool_size=pool_size,
            pool_stride=pool_stride,
            n_filters=n_filters
        )

        # train
        model.train(X_train, y_train, epochs=epochs, lr=lr)

        st.session_state.model = model

        train_acc = model.evaluate(X_train, y_train)
        test_acc  = model.evaluate(X_test, y_test)

        st.success("Training Completed")

        st.write("Train Accuracy:", train_acc)
        st.write("Test Accuracy :", test_acc)


# ─────────────────────────────────────────────────────────────
# PREDICT TAB
# ─────────────────────────────────────────────────────────────
with tab_predict:

    st.subheader("Predict Digit")

    if st.session_state.model is None:
        st.warning("Train model first")
    else:

        model = st.session_state.model

        option = st.radio("Input Method", ["Generate", "Upload"])

        image = None

        # ── Generate sample ──
        if option == "Generate":

            digit = st.selectbox("Digit", list(range(10)))

            if st.button("Predict"):
                image = make_clean_sample(digit)

        # ── Upload image ──
        else:
            st.warning("Upload simple horizontal strip images (1×10 style)")

            invert = st.checkbox("Invert colors (if wrong prediction)")

            uploaded = st.file_uploader("Upload image")

            if uploaded and st.button("Predict Upload"):
                image = parse_uploaded_image(uploaded, invert)

        # ── Prediction ──
        if image is not None:

            st.write("Input image:")
            fig = render_strip(image)
            st.pyplot(fig)

            probs = model.predict_proba(image)
            pred  = int(np.argmax(probs))

            st.write("Predicted Digit:", pred)
            st.write("Confidence:", probs[pred])

#  python -m streamlit run app.py

import numpy as np


# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────

def generate_dataset(n_samples=800, noise_prob=0.02, seed=42):
    """
    Improved dataset with lower noise
    """
    rng = np.random.default_rng(seed)
    images, labels = [], []

    per_class = n_samples // 10

    for digit in range(10):
        for _ in range(per_class):

            img = np.zeros((1, 10), dtype=np.float32)

            if digit > 0:
                pos = rng.choice(10, size=digit, replace=False)
                img[0, pos] = 1.0

            # reduced noise
            if noise_prob > 0:
                mask = rng.random((1, 10)) < noise_prob
                img  = np.abs(img - mask.astype(np.float32))

            images.append(img)
            labels.append(digit)

    images = np.array(images)
    labels = np.array(labels)

    idx = rng.permutation(len(labels))
    return images[idx], labels[idx]


def train_test_split(images, labels, test_ratio=0.2):
    train_idx, test_idx = [], []

    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        np.random.shuffle(cls_idx)

        n_test = int(len(cls_idx) * test_ratio)

        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    return (
        images[train_idx], labels[train_idx],
        images[test_idx],  labels[test_idx]
    )


# ─────────────────────────────────────────────────────────────
# 2. BASIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def convolve(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape

    out_W = W - kW + 1
    out = np.zeros((1, out_W))

    for j in range(out_W):
        out[0, j] = np.sum(image[0, j:j+kW] * kernel[0])

    return out


def relu(x):
    return np.maximum(0, x)


def max_pool(fm, pool_size=2):
    """
    FIX: stride = pool_size (important)
    """
    W = fm.shape[1]

    out_W = W // pool_size
    out = np.zeros((1, out_W))

    for j in range(out_W):
        out[0, j] = np.max(fm[0, j*pool_size:(j+1)*pool_size])

    return out


def fully_connected(x, W, b):
    return W @ x + b


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ─────────────────────────────────────────────────────────────
# 3. CNN MODEL
# ─────────────────────────────────────────────────────────────

class SimpleCNN:

    def __init__(self, kernel_size=3, n_filters=16):

        """
        FIX:
        - Increased filters (better learning)
        """

        self.kernel_size = kernel_size
        self.n_filters = n_filters

        # initialize kernels
        self.kernels = np.random.randn(n_filters, 1, kernel_size) * 0.1

        self._fc_initialised = False


    def _conv_forward(self, image):

        maps = []

        for f in range(self.n_filters):

            fm = convolve(image, self.kernels[f])
            fm = relu(fm)
            fm = max_pool(fm)   # improved pooling

            maps.append(fm)

        return np.array(maps)


    def _init_fc(self, flat_dim):

        self.W1 = np.random.randn(64, flat_dim) * 0.1
        self.b1 = np.zeros(64)

        self.W2 = np.random.randn(10, 64) * 0.1
        self.b2 = np.zeros(10)

        self._fc_initialised = True


    def forward(self, image):

        feat = self._conv_forward(image)

        flat = feat.flatten()

        if not self._fc_initialised:
            self._init_fc(len(flat))

        h1 = relu(fully_connected(flat, self.W1, self.b1))

        logits = fully_connected(h1, self.W2, self.b2)

        probs = softmax(logits)

        cache = (flat, h1, probs)

        return probs, cache


    def _backward(self, cache, label, lr):

        flat, h1, probs = cache

        # error
        dlogits = probs.copy()
        dlogits[label] -= 1

        # FC2
        dW2 = np.outer(dlogits, h1)
        db2 = dlogits

        dh1 = self.W2.T @ dlogits
        dh1[h1 <= 0] = 0

        # FC1
        dW1 = np.outer(dh1, flat)
        db1 = dh1

        # update
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


    def train(self, images, labels, epochs=40, lr=0.01):

        n = len(labels)

        for ep in range(epochs):

            idx = np.random.permutation(n)

            for i in idx:
                probs, cache = self.forward(images[i])
                self._backward(cache, labels[i], lr)


    def predict(self, image):
        probs, _ = self.forward(image)
        return np.argmax(probs)


    def evaluate(self, images, labels):

        correct = 0

        for i in range(len(labels)):
            if self.predict(images[i]) == labels[i]:
                correct += 1

        return correct / len(labels)


# ─────────────────────────────────────────────────────────────
# 4. RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    images, labels = generate_dataset()

    X_train, y_train, X_test, y_test = train_test_split(images, labels)

    model = SimpleCNN()

    model.train(X_train, y_train)

    print("Train Accuracy:", model.evaluate(X_train, y_train))
    print("Test Accuracy :", model.evaluate(X_test, y_test))