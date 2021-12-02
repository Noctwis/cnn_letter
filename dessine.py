import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from keras.models import load_model
from pandas.io.json import json_normalize

model = keras.models.load_model('cnn.h5')
stroke_width = st.sidebar.slider("Stroke width: ", 1, 35, 32)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=300,
    drawing_mode=drawing_mode,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    image_data = image_data.rotate(90)
    image = canvas_result.image_data 
    image1 = image.copy()  
    image1 = image1.astype('uint8')
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    
    image1 = cv2.resize(image1,(28,28))
    st.image(image1)

    image1.resize(1,28,28,1)
    st.title(np.argmax(model.predict(image1)))
    prediction = model.predict(image1)
    if np.argmax(prediction) == 0:
        st.write("A")
    elif np.argmax(prediction) == 1:
        st.write("B")
    elif np.argmax(prediction) == 2:
        st.write("C")
    elif np.argmax(prediction) == 3:
        st.write("D")
    elif np.argmax(prediction) == 4:
        st.write("E")
    elif np.argmax(prediction) == 5:
        st.write("F")
    elif np.argmax(prediction) == 6:
        st.write("G")
    elif np.argmax(prediction) == 7:
        st.write("H")
    elif np.argmax(prediction) == 8:
        st.write("A")
    elif np.argmax(prediction) == 9:
        st.write("J")
    elif np.argmax(prediction) == 10:
        st.write("K")
    elif np.argmax(prediction) == 11:
        st.write("L")
    elif np.argmax(prediction) == 12:
        st.write("M")
    elif np.argmax(prediction) == 13:
        st.write("N")
    elif np.argmax(prediction) == 14:
        st.write("O")
    elif np.argmax(prediction) == 15:
        st.write("P")
    elif np.argmax(prediction) == 16:
        st.write("Q")
    elif np.argmax(prediction) == 17:
        st.write("R")
    elif np.argmax(prediction) == 18:
        st.write("S")
    elif np.argmax(prediction) == 19:
        st.write("T")
    elif np.argmax(prediction) == 20:
        st.write("U")
    elif np.argmax(prediction) == 21:
        st.write("V")
    elif np.argmax(prediction) == 22:
        st.write("W")
    elif np.argmax(prediction) == 23:
        st.write("X")
    elif np.argmax(prediction) == 24:
        st.write("Y")
    elif np.argmax(prediction) == 25:
        st.write("Z")
if canvas_result.json_data is not None:
    st.dataframe(pd.io.json.json_normalize(canvas_result.json_data["objects"]))