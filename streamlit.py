import streamlit as st
import requests
import base64
import io
from PIL import Image
import glob
from base64 import decodebytes
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

##########
##### Set up sidebar.
##########

# Add in location to select image.

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['jpg'],
                                         accept_multiple_files=False)

st.sidebar.write('[Find additional images on Roboflow.](https://universe.roboflow.com/duke-aipi-540-summer-2023/wbc-classification-ih8we)')

##########
##### Set up main app.
##########

## Title.
st.write('# Blood Cell Count Object Detection')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    url = 'https://github.com/changyuhsin1999/WBC-Differential-Learning-Tool/blob/main/data/lymphocyte/LY_409311.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

else:
    # User-selected image.
    image = Image.open(uploaded_file)

## Subtitle.
st.write('### Inferenced Image')

# Convert to JPEG Buffer.
buffered = io.BytesIO()
image.save(buffered, quality=90, format='JPEG')

# Base 64 encode.
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('ascii')