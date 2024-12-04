import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Set page config
st.set_page_config(page_title="EMNIST Letters Recognition App", layout="wide")

def process_emnist_image(image_data):
    """Process EMNIST image by flipping and rotating"""
    # Reshape to 28x28
    image = image_data.reshape(28, 28)
    # Flip vertically (x-axis)
    image = np.flip(image, axis=0)
    # Rotate 90 degrees clockwise
    image = np.rot90(image, k=-1)
    return image

# Load the trained model
@st.cache_resource
def load_letter_model():
    return load_model('letter_recognition_model.h5')

# Load datasets
@st.cache_data
def load_dataset(dataset_type):
    if dataset_type == "Training":
        return pd.read_csv("dataset/train/emnist-letters-train.csv")
    else:
        return pd.read_csv("dataset/test/emnist-letters-test.csv")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Viewer", "Model Testing", "Test Predictions"])

if page == "Dataset Viewer":
    st.title("EMNIST Letters Dataset Viewer")
    
    # Dataset selection
    dataset_type = st.radio("Select Dataset", ["Training", "Test"])
    
    try:
        data = load_dataset(dataset_type)
        st.success(f"Successfully loaded {dataset_type} dataset!")
        
        # Get total number of images
        total_images = len(data)
        st.write(f"Total images in {dataset_type} dataset: {total_images}")
        
        # Image selector
        image_index = st.slider("Select image index", 0, total_images-1, 0)
        
        # Display image
        st.subheader(f"Image {image_index}")
        
        # Get the image data and label
        image_data = data.iloc[image_index, 1:].values
        label = data.iloc[image_index, 0]
        
        # Convert label to letter
        letter = chr(label + 64)
        
        # Process the image
        image = process_emnist_image(image_data)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(0.5, 0.5))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        plt.title(f'Label: {letter} (Class {label})')
        
        # Display the plot
        st.pyplot(fig)
        
        # Display statistics
        st.subheader("Image Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Min pixel value: {image_data.min()}")
            st.write(f"Max pixel value: {image_data.max()}")
        with col2:
            st.write(f"Mean pixel value: {image_data.mean():.2f}")
            st.write(f"Standard deviation: {image_data.std():.2f}")
            
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

elif page == "Model Testing":
    st.title("Model Testing Results")
    
    try:
        # Load test data and model
        test_data = load_dataset("Test")
        model = load_letter_model()
        
        # Prepare test data
        X_test = test_data.iloc[:, 1:].values
        y_test = test_data.iloc[:, 0].values
        X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
        y_test_cat = to_categorical(y_test - 1, num_classes=26)
        
        # Model evaluation
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        st.subheader("Overall Model Performance")
        st.write(f"Test Accuracy: {test_accuracy*100:.2f}%")
        st.write(f"Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = y_test - 1
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        letters = [chr(i + 65) for i in range(26)]
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=letters,
                    yticklabels=letters)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot(fig)
        
        # Per-class accuracy
        st.subheader("Per-class Accuracy")
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Create a bar chart for per-class accuracy
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.bar(letters, class_accuracy * 100)
        plt.title('Accuracy per Letter')
        plt.xlabel('Letter')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        st.pyplot(fig)
        
        # Detailed classification report
        st.subheader("Detailed Classification Report")
        report = classification_report(y_test_classes, y_pred_classes, 
                                    target_names=letters, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)
        
    except Exception as e:
        st.error(f"Error in model testing: {str(e)}")

elif page == "Test Predictions":
    st.title("Test Data Predictions")
    
    try:
        # Load test data and model
        test_data = load_dataset("Test")
        model = load_letter_model()
        
        # Get total number of images
        total_images = len(test_data)
        st.write(f"Total images in test dataset: {total_images}")
        
        # Image selector
        image_index = st.slider("Select image index", 0, total_images-1, 0)
        
        # Get and process the image
        image_data = test_data.iloc[image_index, 1:].values
        true_label = test_data.iloc[image_index, 0]
        true_letter = chr(true_label + 64)
        
        # Process and display the image
        image = process_emnist_image(image_data)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Selected Image")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            plt.title(f'True Label: {true_letter}')
            st.pyplot(fig)
        
        # Prepare for prediction
        img_prepared = image_data.reshape(1, 28, 28, 1) / 255.0
        
        # Make prediction
        prediction = model.predict(img_prepared, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        predicted_letter = chr(predicted_class + 65)
        
        # Display results
        with col2:
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Letter:** {predicted_letter}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
            st.markdown(f"**True Letter:** {true_letter}")
            
            # Show top 3 predictions
            st.subheader("Top 3 Predictions:")
            top3_idx = np.argsort(prediction[0])[-3:][::-1]
            for idx in top3_idx:
                letter = chr(idx + 65)
                conf = prediction[0][idx] * 100
                st.write(f"Letter {letter}: {conf:.2f}%")
                
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.write("""
This app combines three main functionalities:
1. Dataset Viewer: Browse through the EMNIST Letters dataset
2. Model Testing: View detailed model performance metrics
3. Test Predictions: Predict letters from the test dataset
""") 