import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from keras.utils import image_dataset_from_directory
import os
import scipy
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

# List your directories here
directories = ['images_EDA/hamburger', 'images_EDA/hotdog', 'images_EDA/pasta', 'images_EDA/pizza', 'images_EDA/salad']

def main():
    st.title("Image Classification App")
    
    menu = ["Show Images", "Train Model","Show Graph"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        st.subheader("Train Model")
        epochs = st.number_input('Enter number of epochs (min: 5, max: 30)', min_value=5, max_value=30, value=20)
        
        if st.button("Train"):
            model_new, history = train_model(epochs)
            st.write("Model trained successfully!")
    
        if st.button("Plot Error"):
            st.pyplot(plot_error(history))

    elif choice == "Show Images":
        st.subheader("Show Images")
        if st.button("Show"):
            show_images()
    
    elif choice == "Show Graph":
        st.subheader("Show Graph")
        if st.button("Show"):
            show_graph()

def train_model(epochs):
    # Your model definition here
    model_new = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), input_shape = (128, 128, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(28, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Flatten(), # Or, layers.GlobalAveragePooling2D()
        layers.Dense(128, activation="relu"),
        layers.Dense(5, activation="softmax")
    ])

    model_new.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Your data generators here
    train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                           rescale = 1./255,
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_val_datagen.flow_from_directory('images/training',
                                                         subset='training',
                                                         target_size = (128, 128),
                                                         batch_size = 32,
                                                         class_mode = 'categorical') 
    validation_set = train_val_datagen.flow_from_directory('images/training',
                                                           subset='validation',
                                                           target_size = (128, 128),
                                                           batch_size = 32,
                                                           class_mode = 'categorical')

    # Train the model
    history = model_new.fit(training_set,
                            validation_data = validation_set,
                            epochs = epochs)

    return history


def show_images():
    # List your directories here
    directories = ['images_EDA/hamburger', 'images_EDA/hotdog', 'images_EDA/pasta', 'images_EDA/pizza', 'images_EDA/salad']

    # Initialize the figure
    fig, axs = plt.subplots(len(directories), 3, figsize=(10,10))

    # Loop over the directories
    for directory_index, directory in enumerate(directories):
        # Use os.listdir to get all files in the directory
        files = os.listdir(directory)
        
        # Filter the list for files ending in '.jpg', '.jpeg', or '.png'
        images = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Display the first 3 images of each folder
        for image_index, image in enumerate(images[:3]):
            # Open the image
            img = Image.open(os.path.join(directory, image))
            
            # Plot the image
            axs[directory_index, image_index].imshow(img)  # Display the image in its original colors
            axs[directory_index, image_index].axis('off')  # Hide the axis
            axs[directory_index, image_index].set_title(f"{os.path.basename(directory)}:{image}")

    st.write("Bekijk hier 3 afbeeldingen per categorie.")
    # Show the plot in Streamlit
    st.pyplot(fig)


def show_graph():
    # List your directories here
    directories = ['images_EDA/hamburger', 'images_EDA/hotdog', 'images_EDA/pasta', 'images_EDA/pizza', 'images_EDA/salad']

    # Initialize an empty dictionary to hold the counts
    counts = {}

    # Loop over the directories
    for directory in directories:
        # Use os.listdir to get all files in the directory
        files = os.listdir(directory)
        
        # Filter the list for files ending in '.jpg', '.jpeg', or '.png'
        images = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Add the count to the dictionary
        counts[os.path.basename(directory)] = len(images)

    # Create a bar plot of the image counts
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    ax.set_ylabel('Number of Images')

    st.write("Bekijk hier hoeveel afbeeldingen er per categorie zijn.")
    # Show the plot in Streamlit
    st.pyplot(fig)

def plot_error(history):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the figure in Streamlit
    st.pyplot(fig)

if __name__ == "__main__":
    main()
