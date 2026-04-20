import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from cnn_model import (
    generate_dataset,
    split_data,
    build_model,
    train_model,
    evaluate_model,
    predict_digit
)

st.title("CNN Digit Predictor")

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
if st.button("Train Model"):

    images, labels = generate_dataset(800)

    images = images.reshape(-1,1,10,1)

    X_train, X_test, y_train, y_test = split_data(images, labels)

    model = build_model()
    model = train_model(model, X_train, y_train)

    acc = evaluate_model(model, X_test, y_test)

    st.session_state.model = model

    st.success(f"Training Done! Accuracy: {acc:.2f}")


# ─────────────────────────────────────────────
# GENERATE SAMPLE (FIXED)
# ─────────────────────────────────────────────
def generate_sample(digit):
    img = np.zeros((1,10))

    if digit > 0:
        pos = np.random.choice(10, digit, replace=False)
        img[0, pos] = 1

    return img


# ─────────────────────────────────────────────
# DISPLAY (CORRECT VISUAL — BOXES)
# ─────────────────────────────────────────────
def show_image(img):

    fig, ax = plt.subplots(figsize=(8,1))

    for i in range(10):
        val = img[0][i]
        color = "black" if val > 0.5 else "white"

        rect = patches.Rectangle((i,0), 1, 1,
                                 facecolor=color,
                                 edgecolor='gray')

        ax.add_patch(rect)

    ax.set_xlim(0,10)
    ax.set_ylim(0,1)
    ax.set_xticks(range(10))
    ax.set_yticks([])
    ax.set_aspect('equal')

    st.pyplot(fig)


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
if "model" in st.session_state:

    st.subheader("Predict Digit")

    digit = st.selectbox("Choose digit", list(range(10)))

    if st.button("Generate & Predict"):

        img = generate_sample(digit)

        st.write("Input Image:")
        show_image(img)

        pred = predict_digit(st.session_state.model, img)

        st.write("Predicted Digit:", pred)
        st.write("Actual Digit:", digit)
# # """
# # app.py  —  Streamlit front-end for the Custom CNN Digit Predictor (fixed)
# # =========================================================================
# # Run with:
# #     streamlit run app.py

# # Requirements:
# #     pip install streamlit numpy pillow matplotlib
# # """

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as mpatches
# # import streamlit as st
# # from PIL import Image

# # from cnn_model import (
# #     SimpleCNN,
# #     generate_dataset,
# #     train_test_split,
# # )

# # # ─────────────────────────────────────────────────────────────────────────────
# # # PAGE CONFIG
# # # ─────────────────────────────────────────────────────────────────────────────
# # st.set_page_config(page_title="CNN Digit Predictor", page_icon="🔢", layout="wide")

# # st.title("🔢 CNN Digit Predictor  —  pure NumPy")
# # st.markdown(
# #     "All layers (Convolution, Pooling, Padding, Fully-Connected) are "
# #     "**hand-coded in NumPy**. Images are 1×10 binary strips where the "
# #     "number of black cells encodes the digit."
# # )

# # # ─────────────────────────────────────────────────────────────────────────────
# # # SIDEBAR  —  hyper-parameters
# # # ─────────────────────────────────────────────────────────────────────────────
# # st.sidebar.header("⚙️ Hyper-parameters")

# # kernel_size = st.sidebar.slider("Kernel size (width)", 1, 5, 3)
# # pool_size   = st.sidebar.slider("Pool size (width)",   1, 4, 2)
# # pool_stride = st.sidebar.slider("Pool stride",         1, 3, 1)
# # n_filters   = st.sidebar.slider("Number of filters",   2, 16, 8)
# # epochs      = st.sidebar.slider("Training epochs",     5, 80, 40, step=5)
# # lr          = st.sidebar.select_slider(
# #                   "Learning rate",
# #                   options=[0.001, 0.005, 0.01, 0.05, 0.1],
# #                   value=0.01)
# # n_samples   = st.sidebar.slider("Dataset size", 200, 1200, 800, step=100)
# # noise_prob  = st.sidebar.slider("Cell noise probability", 0.0, 0.20, 0.08, step=0.01,
# #                                  help="Randomly flips cells in the image to make the task harder")

# # # ─────────────────────────────────────────────────────────────────────────────
# # # HELPERS
# # # ─────────────────────────────────────────────────────────────────────────────

# # def render_strip(image_1x10: np.ndarray, title: str = "") -> plt.Figure:
# #     fig, ax = plt.subplots(figsize=(6, 1.2))
# #     row = image_1x10[0]
# #     for j, val in enumerate(row):
# #         colour = "black" if val > 0.5 else "white"
# #         rect = mpatches.FancyBboxPatch(
# #             (j + 0.05, 0.05), 0.9, 0.9,
# #             boxstyle="round,pad=0.05",
# #             linewidth=1.5, edgecolor="gray", facecolor=colour,
# #         )
# #         ax.add_patch(rect)
# #     ax.set_xlim(0, 10)
# #     ax.set_ylim(0, 1)
# #     ax.set_aspect("equal")
# #     ax.axis("off")
# #     if title:
# #         ax.set_title(title, fontsize=11)
# #     plt.tight_layout()
# #     return fig


# # def make_clean_sample(digit: int, rng=None) -> np.ndarray:
# #     """Generate a NOISE-FREE 1×10 image for display/prediction."""
# #     if rng is None:
# #         rng = np.random.default_rng()
# #     img = np.zeros((1, 10), dtype=np.float32)
# #     if digit > 0:
# #         pos = rng.choice(10, size=digit, replace=False)
# #         img[0, pos] = 1.0
# #     return img


# # def parse_uploaded_image(uploaded_file):
# #     try:
# #         img = Image.open(uploaded_file).convert("L")
# #         img = img.resize((10, 1), Image.NEAREST)
# #         arr = np.array(img, dtype=np.float32) / 255.0   # (1, 10)
# #         arr = (arr < 0.5).astype(np.float32)             # dark → 1.0
# #         return arr
# #     except Exception as e:
# #         st.error(f"Could not process image: {e}")
# #         return None


# # # ─────────────────────────────────────────────────────────────────────────────
# # # SESSION STATE
# # # ─────────────────────────────────────────────────────────────────────────────
# # for key in ["model", "train_acc", "test_acc", "loss_curve"]:
# #     if key not in st.session_state:
# #         st.session_state[key] = None if key != "loss_curve" else []

# # # ─────────────────────────────────────────────────────────────────────────────
# # # TABS
# # # ─────────────────────────────────────────────────────────────────────────────
# # tab_train, tab_predict, tab_explain = st.tabs(
# #     ["🏋️ Train Model", "🔍 Predict Digit", "📖 How It Works"]
# # )

# # # ══════════════════════════════════════════════════════════════════════════════
# # # TAB 1 — TRAIN
# # # ══════════════════════════════════════════════════════════════════════════════
# # with tab_train:
# #     st.subheader("Train the CNN")
# #     st.markdown("Adjust hyper-parameters in the sidebar, then click **Train**.")

# #     if st.button("🚀 Train Model", type="primary"):

# #         # Build dataset
# #         with st.spinner("Generating dataset …"):
# #             images, labels = generate_dataset(n_samples=n_samples,
# #                                               noise_prob=noise_prob)
# #             X_train, y_train, X_test, y_test = train_test_split(images, labels)

# #         st.info(f"Train: {len(y_train)} samples  |  Test: {len(y_test)} samples  "
# #                 f"|  Noise: {noise_prob:.0%} per cell")

# #         # Build model
# #         model = SimpleCNN(
# #             kernel_size=kernel_size,
# #             pool_size=pool_size,
# #             pool_stride=pool_stride,
# #             n_filters=n_filters,
# #         )

# #         # Train epoch-by-epoch so we can show a live progress bar
# #         progress  = st.progress(0, text="Starting …")
# #         loss_log  = []
# #         rng_train = np.random.default_rng(99)
# #         n         = len(y_train)

# #         for ep in range(1, epochs + 1):
# #             idx   = rng_train.permutation(n)
# #             total = 0.0
# #             for i in idx:
# #                 probs, cache = model.forward(X_train[i])
# #                 total += -np.log(probs[y_train[i]] + 1e-9)
# #                 model._backward(cache, y_train[i], lr)
# #             loss_log.append(total / n)
# #             progress.progress(ep / epochs,
# #                               text=f"Epoch {ep}/{epochs}  loss={loss_log[-1]:.4f}")

# #         progress.empty()

# #         # Store in session state
# #         st.session_state.model      = model
# #         st.session_state.train_acc  = model.evaluate(X_train, y_train)
# #         st.session_state.test_acc   = model.evaluate(X_test,  y_test)
# #         st.session_state.loss_curve = loss_log
# #         # Keep test data for per-digit breakdown
# #         st.session_state.X_test  = X_test
# #         st.session_state.y_test  = y_test

# #         st.success("Training complete!")

# #     # ── Metrics & loss curve ───────────────────────────────────────────────
# #     if st.session_state.model is not None:
# #         col1, col2 = st.columns(2)
# #         col1.metric("Train Accuracy", f"{st.session_state.train_acc*100:.1f}%")
# #         col2.metric("Test Accuracy",  f"{st.session_state.test_acc*100:.1f}%")

# #         fig, ax = plt.subplots(figsize=(7, 3))
# #         ax.plot(st.session_state.loss_curve, color="#2563EB", linewidth=2)
# #         ax.set_xlabel("Epoch")
# #         ax.set_ylabel("Cross-Entropy Loss")
# #         ax.set_title("Training Loss Curve")
# #         ax.grid(True, alpha=0.3)
# #         st.pyplot(fig)
# #         plt.close(fig)

# #         # Per-digit accuracy breakdown
# #         with st.expander("Per-digit test accuracy"):
# #             X_test = st.session_state.X_test
# #             y_test = st.session_state.y_test
# #             model  = st.session_state.model
# #             accs   = []
# #             for d in range(10):
# #                 idx = np.where(y_test == d)[0]
# #                 if len(idx) == 0:
# #                     accs.append(0.0)
# #                     continue
# #                 acc = sum(model.predict(X_test[i]) == d for i in idx) / len(idx)
# #                 accs.append(acc)

# #             fig2, ax2 = plt.subplots(figsize=(7, 3))
# #             bars = ax2.bar(range(10), [a*100 for a in accs], color="#2563EB")
# #             ax2.set_xticks(range(10))
# #             ax2.set_xlabel("Digit")
# #             ax2.set_ylabel("Accuracy (%)")
# #             ax2.set_title("Per-digit Test Accuracy")
# #             ax2.set_ylim(0, 110)
# #             for i, a in enumerate(accs):
# #                 ax2.text(i, a*100 + 1, f"{a*100:.0f}%", ha="center", fontsize=8)
# #             st.pyplot(fig2)
# #             plt.close(fig2)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # TAB 2 — PREDICT
# # # ══════════════════════════════════════════════════════════════════════════════
# # with tab_predict:
# #     st.subheader("Predict a Digit")

# #     if st.session_state.model is None:
# #         st.warning("⚠️ Please train the model first (Tab 1).")
# #     else:
# #         model = st.session_state.model

# #         input_method = st.radio(
# #             "Input method",
# #             ["Generate clean sample", "Upload image"],
# #             horizontal=True,
# #         )

# #         image_to_predict = None

# #         if input_method == "Generate clean sample":
# #             digit_choice = st.selectbox("Choose digit:", list(range(10)))
# #             if st.button("Generate & Predict"):
# #                 image_to_predict = make_clean_sample(digit_choice)

# #         else:
# #             st.markdown(
# #                 "Upload any image. It will be resized to 1×10 and thresholded. "
# #                 "Dark pixels → black cell (count as 1)."
# #             )
# #             uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","bmp"])
# #             if uploaded and st.button("Predict from upload"):
# #                 image_to_predict = parse_uploaded_image(uploaded)

# #         # ── Run prediction ────────────────────────────────────────────────
# #         if image_to_predict is not None:
# #             st.markdown("---")
# #             st.markdown("**Input image (1×10 strip):**")
# #             n_black = int(image_to_predict.sum())
# #             fig = render_strip(image_to_predict,
# #                                title=f"Black cells: {n_black}")
# #             st.pyplot(fig)
# #             plt.close(fig)

# #             probs = model.predict_proba(image_to_predict)
# #             pred  = int(np.argmax(probs))
# #             conf  = probs[pred] * 100

# #             st.markdown(f"## Predicted digit: **{pred}**  &nbsp; confidence: `{conf:.1f}%`")

# #             # Probability bar chart
# #             fig2, ax2 = plt.subplots(figsize=(7, 3))
# #             bars = ax2.bar(range(10), probs * 100, color="#94A3B8")
# #             bars[pred].set_color("#16A34A")
# #             ax2.set_xticks(range(10))
# #             ax2.set_xlabel("Digit")
# #             ax2.set_ylabel("Probability (%)")
# #             ax2.set_title("Class Probabilities")
# #             ax2.set_ylim(0, 110)
# #             for i, p in enumerate(probs):
# #                 ax2.text(i, p*100 + 1, f"{p*100:.1f}", ha="center", fontsize=8)
# #             st.pyplot(fig2)
# #             plt.close(fig2)

# #         # ── Batch: one sample per digit ───────────────────────────────────
# #         with st.expander("Show one clean sample per digit (0–9)"):
# #             rng2 = np.random.default_rng(7)
# #             for d in range(10):
# #                 img  = make_clean_sample(d, rng2)
# #                 pred = model.predict(img)
# #                 prob = model.predict_proba(img)[pred] * 100
# #                 ok   = "✅" if pred == d else "❌"
# #                 fig  = render_strip(img,
# #                         title=f"True={d}  |  Predicted={pred} ({prob:.1f}%)  {ok}")
# #                 st.pyplot(fig)
# #                 plt.close(fig)


# # # ══════════════════════════════════════════════════════════════════════════════
# # # TAB 3 — HOW IT WORKS
# # # ══════════════════════════════════════════════════════════════════════════════
# # with tab_explain:
# #     st.subheader("Architecture & Implementation Details")

# #     # Compute output dims dynamically
# #     after_conv = (10 - kernel_size) // 1 + 1
# #     after_pool = max(1, (after_conv - pool_size) // pool_stride + 1) if after_conv >= pool_size else 1
# #     flat_dim   = n_filters * after_pool

# #     st.markdown(
# #         f"""
# # ### Dataset
# # - **{n_samples} images** of size **1×10** (one row, ten columns).
# # - Cell value **0.0 = white**, **1.0 = black**.
# # - Digit *k* → exactly *k* randomly chosen cells are black.
# # - **Noise**: each cell independently flipped with probability `{noise_prob}` — makes the task genuinely hard.
# # - **Stratified 80-20 train/test split** to preserve class balance.

# # ---
# # ### Custom Layer Functions (pure NumPy)

# # | Function | Description |
# # |---|---|
# # | `pad(image, pad_width)` | Adds zero-border on all sides |
# # | `convolve(image, kernel, stride, padding)` | Slides kernel; dot-product at each position |
# # | `relu(x)` | Clips negatives to 0 |
# # | `max_pool(feat, pool_size, stride)` | Takes max in each window |
# # | `avg_pool(feat, pool_size, stride)` | Takes mean in each window |
# # | `fully_connected(x, W, b)` | Linear transform: W @ x + b |
# # | `softmax(x)` | Converts logits → probabilities |

# # ---
# # ### Forward Pass with Current Settings

# # ```
# # Input      (1, 10)
# #   │
# #   Conv      {n_filters} filters,  kernel width = {kernel_size}
# #   ReLU
# #   MaxPool   window = {pool_size},  stride = {pool_stride}
# #   │
# #   Output width after conv : ({10} − {kernel_size}) / 1 + 1 = {after_conv}
# #   Output width after pool : ({after_conv} − {pool_size}) / {pool_stride} + 1 = {after_pool}
# #   │
# #   Flatten   {n_filters} × {after_pool} = {flat_dim} features
# #   │
# #   FC1       {flat_dim} → 64  (ReLU)
# #   FC2       64 → 10           (Softmax)
# # ```

# # ---
# # ### Training Details
# # - **SGD** (stochastic gradient descent), one sample per update.
# # - **Cross-entropy loss**.
# # - **He initialisation** — weights scaled by √(2 / fan_in) to prevent saturation.
# # - **Gradient clipping** at ±5 to prevent exploding gradients.
# # - Backprop through both FC layers.

# # ---
# # ### Why 100% was wrong (and what we fixed)
# # | Problem | Fix |
# # |---|---|
# # | No noise → trivial counting task | Added `noise_prob` cell flips |
# # | Small random init → softmax saturates fast | He initialisation |
# # | Unbounded gradients → one class wins | Gradient clipping |
# # | FC too narrow (32 units) | Widened to 64 units |
# #         """
# #     )
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import streamlit as st
# from PIL import Image

# from cnn_model import SimpleCNN, generate_dataset, train_test_split


# # ─────────────────────────────────────────────────────────────
# # PAGE SETUP
# # ─────────────────────────────────────────────────────────────
# st.set_page_config(page_title="CNN Digit Predictor", page_icon="🔢", layout="wide")

# st.title("🔢 CNN Digit Predictor")

# st.markdown("Train a CNN and predict digits from 1×10 binary images.")


# # ─────────────────────────────────────────────────────────────
# # SIDEBAR (FIXED PARAMETERS)
# # ─────────────────────────────────────────────────────────────
# st.sidebar.header("⚙️ Hyperparameters")

# kernel_size = st.sidebar.slider("Kernel size", 1, 5, 3)
# pool_size   = st.sidebar.slider("Pool size", 1, 4, 2)

# # ✅ FIX: force stride = pool_size
# pool_stride = pool_size

# n_filters   = st.sidebar.slider("Filters", 4, 32, 16)
# epochs      = st.sidebar.slider("Epochs", 5, 60, 40)
# lr          = st.sidebar.select_slider(
#     "Learning rate",
#     options=[0.001, 0.005, 0.01],
#     value=0.01
# )

# n_samples   = st.sidebar.slider("Dataset size", 200, 1200, 800)
# noise_prob  = st.sidebar.slider("Noise", 0.0, 0.1, 0.02)


# # ─────────────────────────────────────────────────────────────
# # HELPERS
# # ─────────────────────────────────────────────────────────────

# def render_strip(image):
#     fig, ax = plt.subplots(figsize=(6, 1))
#     row = image[0]

#     for j, val in enumerate(row):
#         color = "black" if val > 0.5 else "white"
#         rect = mpatches.Rectangle((j, 0), 1, 1, facecolor=color)
#         ax.add_patch(rect)

#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 1)
#     ax.axis("off")

#     return fig


# def make_clean_sample(digit):
#     img = np.zeros((1, 10))
#     if digit > 0:
#         pos = np.random.choice(10, digit, replace=False)
#         img[0, pos] = 1
#     return img


# def parse_uploaded_image(uploaded_file, invert=False):
#     """
#     FIXED:
#     - Better thresholding
#     - Optional inversion
#     """

#     img = Image.open(uploaded_file).convert("L")

#     # resize to 1×10
#     img = img.resize((10, 1))

#     arr = np.array(img) / 255.0

#     # adaptive threshold
#     threshold = np.mean(arr)
#     arr = (arr < threshold).astype(np.float32)

#     # optional inversion
#     if invert:
#         arr = 1 - arr

#     return arr


# # ─────────────────────────────────────────────────────────────
# # SESSION STATE
# # ─────────────────────────────────────────────────────────────
# if "model" not in st.session_state:
#     st.session_state.model = None


# # ─────────────────────────────────────────────────────────────
# # TABS
# # ─────────────────────────────────────────────────────────────
# tab_train, tab_predict = st.tabs(["Train", "Predict"])


# # ─────────────────────────────────────────────────────────────
# # TRAIN TAB
# # ─────────────────────────────────────────────────────────────
# with tab_train:

#     st.subheader("Train Model")

#     if st.button("Train Model"):

#         # generate dataset
#         images, labels = generate_dataset(n_samples, noise_prob)

#         X_train, y_train, X_test, y_test = train_test_split(images, labels)

#         # create model
#         model = SimpleCNN(
#             kernel_size=kernel_size,
#             pool_size=pool_size,
#             pool_stride=pool_stride,
#             n_filters=n_filters
#         )

#         # train
#         model.train(X_train, y_train, epochs=epochs, lr=lr)

#         st.session_state.model = model

#         train_acc = model.evaluate(X_train, y_train)
#         test_acc  = model.evaluate(X_test, y_test)

#         st.success("Training Completed")

#         st.write("Train Accuracy:", train_acc)
#         st.write("Test Accuracy :", test_acc)


# # ─────────────────────────────────────────────────────────────
# # PREDICT TAB
# # ─────────────────────────────────────────────────────────────
# with tab_predict:

#     st.subheader("Predict Digit")

#     if st.session_state.model is None:
#         st.warning("Train model first")
#     else:

#         model = st.session_state.model

#         option = st.radio("Input Method", ["Generate", "Upload"])

#         image = None

#         # ── Generate sample ──
#         if option == "Generate":

#             digit = st.selectbox("Digit", list(range(10)))

#             if st.button("Predict"):
#                 image = make_clean_sample(digit)

#         # ── Upload image ──
#         else:
#             st.warning("Upload simple horizontal strip images (1×10 style)")

#             invert = st.checkbox("Invert colors (if wrong prediction)")

#             uploaded = st.file_uploader("Upload image")

#             if uploaded and st.button("Predict Upload"):
#                 image = parse_uploaded_image(uploaded, invert)

#         # ── Prediction ──
#         if image is not None:

#             st.write("Input image:")
#             fig = render_strip(image)
#             st.pyplot(fig)

#             probs = model.predict_proba(image)
#             pred  = int(np.argmax(probs))

#             st.write("Predicted Digit:", pred)
#             st.write("Confidence:", probs[pred])

# #  python -m streamlit run app.py