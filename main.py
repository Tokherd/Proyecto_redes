import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
#
# Titulo
st.set_page_config(layout="centered",page_title="ACGAN MNIST generator")

st.title("Generador de digitos MNIST con ACGAN")
st.write("Selecciona un difito para generar imagenes")
@st.cache_resource# cargar el modelo
def load_acgan_model():
    model_path= "acgan_mnist.h5"
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")


trained_generator=load_acgan_model()
latent_size=100
num_classes=10

st.sidebar.header("opciones de generación")
selected_label=st.sidebar.slider(
    "selecciona un digito a generar", 
    min_value=0,
    max_value=9,
    value=0,
    step=1
)

def generate_and_plot_acgan(generator,latent_size, num_classes,class_label):
    st.subheader(f"Generando imagenes para el digito: {class_label}")
    noise_input = np.random.uniform(-1.0, 1.0, size=(16, latent_size))
    noise_class_labels=np.ones(16,dtype='int32')*class_label
    noise_class_input = noise_class_labels.reshape(-1, 1)  

    # predict images
    with st.spinner("Generando imagenes..."):
        images = generator.predict([noise_input, noise_class_input],verbose=0)

    fig,ax=plt.subplots(figsize=(8,8))
    num_images= images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images / rows))
    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {noise_class_labels[i]}")
        plt.axis('off')
    plt.tight_layout()  # Ajustar espaciado
    st.pyplot(fig)  # Mostrar la figura en Streamlit
    plt.close(fig)  # Cerrar la figura para liberar memoria

# correr el modelo cuando se cargue la app o cambie el valor del slider

if st.button("Generar imagenes") or st.session_state.get("initial_run", True):

    generate_and_plot_acgan(trained_generator, latent_size, num_classes, selected_label)
    st.session_state["initial_run"] = False  # Marcar que la app ya se ha ejecutado una vez

st.markdown("---")
st.write("Este modelo fue entrenado con el dataset MNIST y utiliza ACGAN para generar imagenes de digitos.")