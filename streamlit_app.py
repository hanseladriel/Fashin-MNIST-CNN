import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import struct

model = tf.keras.models.load_model('model2.keras')
class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num, rows, cols)
        return images / 255.0  # normalize

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

x_test = load_images('data/t10k-images-idx3-ubyte')
y_test = load_labels('data/t10k-labels-idx1-ubyte')

st.title("Fashion MNIST Classifier")

option = st.radio("Choose input type:", ("Select from test set", "Upload an image"))

if option == "Select from test set":
    index = st.slider("Select image index", 0, len(x_test) - 1, 0)
    image = x_test[index]
    st.image(image, caption="Selected Test Image", width=200)
    
    input_img = image.reshape(1, 28, 28, 1)
elif option == "Upload an image":
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert('L').resize((28, 28))
        st.image(img, caption="Uploaded Image", width=200)
        image = np.array(img) / 255.0
        input_img = image.reshape(1, 28, 28, 1)

if 'input_img' in locals():
    pred = model.predict(input_img)
    label = class_names[np.argmax(pred)]
    st.success(f"Predicted Class: **{label}**")