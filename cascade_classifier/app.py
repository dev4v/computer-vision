import pandas as pd
import cv2
import streamlit as st
import numpy as np
from PIL import Image
import cv2.data

st.title("eye detection using cascade classifer")

# Load the Haar cascade xml
cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(cascade_path)

# File uploader
img_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
if img_file is not None:
    # Read into PIL, convert to BGR NumPy array
    pil_image = Image.open(img_file)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    #  Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw bounding boxes
    for (x, y, w, h) in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #Convert back to RGB and display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Detected eyes", use_column_width=True)




