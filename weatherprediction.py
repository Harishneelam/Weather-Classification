import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from IPython.core.display import HTML


st.set_page_config(page_title="Weather Prediction")

model_path = 'model_main.h5'
model_main = load_model(model_path)

model_path = 'model_cold.h5'
model_cold = load_model(model_path)

model_path = 'model_cold_wt.h5'
model_cold_wt = load_model(model_path)

model_path = 'model_dusty.h5'
model_dusty = load_model(model_path)

model_path = 'model_rainy.h5'
model_rainy = load_model(model_path)

def predict_weather(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100), resample=Image.LANCZOS).convert('RGB')
    img_arr = np.asarray(img).reshape(1, 100, 100, 3)

    x_train_mean = 131.7835376361996 
    x_train_std = 64.82395884131256
    img_main = (img_arr - x_train_mean) / x_train_std

    Types = ['Cold', 'Dusty', 'Rainy']
    result = model_main.predict(img_main, verbose=0)
    max_prob = np.argmax(result[0])
    string = 'There is <b>' + str(round(max(result[0])*100,2)) + '%</b> chance that the Image has <b>' + str(Types[max_prob]).capitalize() + '</b> Weather.'
    st.write(HTML(string), unsafe_allow_html=True)

    if max_prob == 0:
        x_train_mean = 133.95774408217363 
        x_train_std = 66.46460631111488 
        img_cold = (img_arr - x_train_mean) / x_train_std

        Types = ['dew', 'frost', 'glaze', 'rime', 'snow']
        result = model_cold.predict(img_cold, verbose=0)
        max_prob = np.argmax(result[0])
        string = 'It is <b>' + str(Types[max_prob]).capitalize() + '</b> with <b>' + str(round(max(result[0])*100,2)) + '%</b> similarity score.'
        st.write(HTML(string), unsafe_allow_html=True)

        Types = ['Potentially Hazardous', 'Safe']
        result = model_cold_wt.predict(img_cold, verbose=0)
        max_prob = np.argmax(result[0])

        string = 'It is <b>' + str(Types[max_prob]).capitalize() + '</b> out there.<b>'
        st.write(HTML(string), unsafe_allow_html=True)

    
    elif(max_prob == 1):
        x_train_mean = 139.81614348765433 
        x_train_std = 58.56408739097014
        img_dusty = (img_arr - x_train_mean) / x_train_std
        Types = ['fogsmog', 'sandstorm']
        result = model_dusty.predict(img_dusty, verbose = 0)
        max_prob = np.argmax(result[0])
        string = 'It is <b>' + str(Types[max_prob]).capitalize() + '</b> with <b>' + str(round(max(result[0])*100,2)) + '%</b> similarity score.'
        st.write(HTML(string), unsafe_allow_html=True)
    
        if(max_prob == 0):
            string = 'It is <b> Safe </b> out there, but be careful though.<b>'
            st.write(HTML(string), unsafe_allow_html=True)
        else:
            string = 'It is <b> Dangerous</b>.<b>'
            st.write(HTML(string), unsafe_allow_html=True)
    
    else:
        x_train_mean = 118.63095932671082
        x_train_std = 65.4575389955544
        img_rainy = (img_arr - x_train_mean) / x_train_std
        Types = ['hail', 'lightning', 'rain', 'rainbow']
        result = model_rainy.predict(img_rainy, verbose = 0)
        max_prob = np.argmax(result[0])
        string = 'It is <b>' + str(Types[max_prob]).capitalize() + '</b> with <b>' + str(round(max(result[0])*100,2)) + '%</b> similarity score.'
        st.write(HTML(string), unsafe_allow_html=True)

        if(max_prob == 3):
            string = 'It is <b> Safe </b> out there.<b>'
            st.write(HTML(string), unsafe_allow_html=True)
        elif(max_prob == 1):
            string = 'It is <b> Dangerous</b>.<b>'
            st.write(HTML(string), unsafe_allow_html=True)
        else:
            string = 'It is <b>Potentially Hazardous</b>, be careful.<b>'
            st.write(HTML(string), unsafe_allow_html=True)


import streamlit as st


st.title("Weather Prediction")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting weather...")
    predict_weather(uploaded_file)

