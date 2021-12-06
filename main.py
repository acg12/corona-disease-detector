import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow_hub as hub
from PIL import Image
st.title('Corona Disease Detector')


@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model('model.h5', compile=False,
                       custom_objects={'KerasLayer': hub.KerasLayer})
    return model


# A dictionary of class label and batik name
batik_dict = {
    0: 'Batik Bali',
    1: 'Batik Betawi',
    2: 'Batik Cendrawasih',
    3: 'Batik Dayak',
    4: 'Batik Geblek Renteng',
    5: 'Batik Ikat Celup',
    6: 'Batik Insang',
    7: 'Batik Kawung',
    8: 'Batik Lasem',
    9: 'Batik Megamendung',
    10: 'Batik Pala',
    11: 'Batik Parang',
    12: 'Batik Poleng',
    13: 'Batik Sekar Jagad',
    14: 'Batik Tambal'
}


def predict(img):
    img = img.resize((224, 224))  # resize
    img = img_to_array(img)  # convert to array
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # 1d array
    pred = np.argmax(model.predict(img))  # predict image, get class label
    return pred


if __name__ == '__main__':
    st.write('This is a prototype of the Corona Disease Detector.')
    st.write('Please upload an x-ray to diagnose.')
    st.write('The classification result will be displayed below.')

    model = load_trained_model()
    uploaded_file = st.file_uploader(
        'Upload an image of batik', type=['jpg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        class_label = predict(image)
        st.info('This is {}'.format(batik_dict[class_label]))
        st.image(image, caption='Uploaded Image')
