import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import cv2

st.set_page_config(page_title="Doodle Digit Recognizer", layout="centered")
st.title("🎨 Draw a Digit (0–9) and Let the CNN Predict It!")

# Load + train model (cached)
@st.cache_resource
def train_model():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    return model

model = train_model()

# Draw canvas
st.subheader("✏️ Draw a digit below:")
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Preprocess: convert to 28x28 grayscale
    gray = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    scaled = resized.astype("float32") / 255.0
    reshaped = scaled.reshape(1, 28, 28, 1)

    prediction = model.predict(reshaped)[0]
    predicted_class = np.argmax(prediction)

    st.subheader("🧠 Prediction:")
    st.write(f"### Predicted Digit: `{predicted_class}`")
    st.bar_chart(prediction)
