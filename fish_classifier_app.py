import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pandas as pd 
import base64
import time

# --- Background Image Setup ---
BACKGROUND_IMAGE_PATH = 'background.jpg' 

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    
    # Check if the file exists before trying to load it
    if not os.path.exists(png_file):
        st.error(f"Background image file not found: {png_file}. Please ensure it's in the same directory as the app.")
        return

    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Define target image size (must match what your model was trained on)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Define the classes (must match the order your model was trained on)
CLASSES = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream','fish sea_food hourse_mackerel','fish sea_food red_mullet','fish sea_food red_sea_bream','fish sea_food sea_bass',\
           'fish sea_food shrimp','fish sea_food striped_red_mullet','fish sea_food trout']
NUM_CLASSES = len(CLASSES)

# --- Model Loading ---
MODEL_PATH = 'best_inceptionv3_model.h5'

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    """Loads the pre-trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file not found at {MODEL_PATH}")
        st.stop()                                               # Stop the app if the model isn't found
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Streamlit Application ---

st.set_page_config(page_title="ScaleScam: Fish Classifier", layout="centered")

# Set the background image
set_background(BACKGROUND_IMAGE_PATH)

st.title("🐟 ScaleScan: Fish Classifier")
st.write("Upload an image of a fish, and I'll predict its species!")

# Handle example image selection
uploaded_file_data = None
uploaded_file_name = None
if 'uploaded_file_data' in st.session_state and st.session_state.uploaded_file_data:
    uploaded_file = st.session_state.uploaded_file_data
    uploaded_file_name = st.session_state.uploaded_file_name
    
    # Clear session state after use to prevent re-processing on subsequent reruns
    del st.session_state.uploaded_file_data
    del st.session_state.uploaded_file_name

else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
   
    # Display the uploaded image
    img = Image.open(uploaded_file).convert('RGB')      # Ensure image is RGB
    st.image(img, caption='Uploaded Image', use_container_width=True)
    st.write("")

    # --- Loading Indicator ---
    with st.spinner('Classifying image...'):
        time.sleep(1)                                   
        
        # --- Preprocessing the image ---
        img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # --- Prediction ---
        predictions = model.predict(img_array)[0]
        predicted_class_idx = np.argmax(predictions)
        predicted_class_name = CLASSES[predicted_class_idx]
        confidence = predictions[predicted_class_idx] * 100

    st.subheader(f"Prediction: **{predicted_class_name}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # --- Top 3 Predictions ---
    st.write("---")
    st.subheader("Top 3 Class Confidence Scores:")
    
    # Create a DataFrame for all predictions and sort by confidence
    all_predictions_df = pd.DataFrame({
        'Class': CLASSES,
        'Confidence': predictions
    })
    all_predictions_df = all_predictions_df.sort_values(by='Confidence', ascending=False)
    st.dataframe(all_predictions_df.head(3).style.format({'Confidence': '{:.2%}'}))

    st.write("---")
    st.subheader("All Class Confidence Scores (Detailed):")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Confidence', y='Class', data=all_predictions_df, palette='viridis', ax=ax)
    ax.set_title('Confidence Scores for Each Class')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Fish Class')
    st.pyplot(fig)

    # --- User Feedback Mechanism ---
    st.write("---")
    st.subheader("Was this prediction correct?")
    col1, col2 = st.columns(2)

    feedback_file = "feedback_log.txt"

    def log_feedback(image_name, predicted_class, true_class, confidence, correct):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(feedback_file, "a") as f:
            f.write(f"{timestamp}, Image: {image_name}, Predicted: {predicted_class}, \
                    Confidence: {confidence:.2f}%, Correct: {correct}, True_Class: {true_class}\n")
        st.success("Thank you for your feedback!")

    if col1.button("Yes, it's correct!"):
        log_feedback(uploaded_file_name if uploaded_file_name else "Uploaded_Image", predicted_class_name, predicted_class_name, confidence, True)
    if col2.button("No, it's incorrect."):
        log_feedback(uploaded_file_name if uploaded_file_name else "Uploaded_Image", \
                     predicted_class_name, "Unknown/Incorrect", confidence, False)


else:
    st.info("Please upload an image or try an example to get a fish species prediction.")

st.markdown("--------")
st.markdown("Built with Streamlit and TensorFlow")