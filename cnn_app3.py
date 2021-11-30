import streamlit as st
import keras
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
st.title("Image Classification")
st.header("Brest cancer Classification Example")
st.text("Upload a brest cancer Image for image classification as cancer or no-cancer")

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = tensorflow.keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 75, 75, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (75, 75)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_image_array  # (Not sure if this is needed, but gives an error!!!)

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability
    


uploaded_file = st.file_uploader("Choose photo ...", type=["jpg", "png"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='Uploaded cancer.', use_column_width=True)
  st.write("")
  st.write("Classifying...")
  label = teachable_machine_classification(image, 'my_model.hdf5') # Name of the model from Teachablemachine
  if np.argmax(prediction) == 0:
      st.write("Not sick")
  elif np.argmax(prediction) == 1:
      st.write("Sick")
  else:
      st.write("")


