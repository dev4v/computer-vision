import pandas as pd
import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("eye detection using cascade classifer")

img=st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if img is not None:
    # Use PIL to open the image
    pil_image = Image.open(img)

    # Convert to a NumPy array
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  

# Load the Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#gray scale convert  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#detect eyes 
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=5)
#draw bounding boxes
    for x, y, w, h in eyes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    st.image(image_rgb)




