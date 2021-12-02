import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import pandas as pd
#import cv2

def load_checkpoint(filepath):  # loading the pretrained weights
    model = EMNIST()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()

    return model

model = load_checkpoint('model.pth')

def import_and_predict(image_data, model):
    
        size = (28,28)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.rotate(90)
        image = image.convert('RGB')
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        image2_reshape = image.reshape(-1,28,28,1)

        prediction = model.predict(image2_reshape)
        res=np.argmax(prediction,axis=1) 
        
        
        return res 
#mettre prediction a la place de res pour avoir les pourcentages

model = load_checkpoint('model.pth')

st.write("""
         # Letter prédiction
         """
         )

st.write("This is a simple image classification web app to predict letter")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file(jpg or png)")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
from numpy.core.defchararray import title
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from torch_utils import transform_image,  get_prediction
from PIL import Image
import numpy as np
import pandas as pd
import time

#st.set_page_config(
#    page_title="Handwritten Letters Classifier",
#    page_icon=":pencil:",
#)


hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("Handwritten Letters Classifier")


def predict(image):
    image = Image.fromarray((image[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    tensor = transform_image(image)
    prediction = get_prediction(tensor)
    return prediction


def np_to_df(outputs):  # Create a 2D array for the dataframe instead of a 1D array
    length = outputs.shape[0]  # Total outputs
    arr = []
    for pos in range(0, length):
        line = [0]*26
        line[pos] = outputs[pos]
        arr.append(line)
    return arr


# Specify brush parameters and drawing mode
stroke_width = st.sidebar.slider("Stroke width: ", 1, 100, 25)


# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#fff",
    background_color="#000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

result = st.button("Predict")

if canvas_result.image_data is not None and result:
    outputs = predict(canvas_result.image_data)
    ind_max = np.where(outputs == max(outputs))[
        0][0]  # Index of the max element
    # Converting index to equivalent letter
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.markdown("<h3 style = 'text-align: center;'>Prediction : {}<h3>".format(
        chr(97 + ind_max)), unsafe_allow_html=True)
    chart_data = pd.DataFrame(np_to_df(outputs), index=[
                              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], columns=[
                              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
    st.bar_chart(chart_data)
    st.balloons()
    
