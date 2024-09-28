import streamlit as st
from model import train_and_predict
from data import prepare_data

# Load the dataset
train_folder = '/data'

# Prepare data for training
x_train, x_test, x_holdout, no_of_classes, img_width, img_height, batch_size = prepare_data(train_folder)

# Streamlit app
st.title("Indian Sign Language Classification")

# Train the model automatically (for 4 epochs) without asking the user
if "model_trained" not in st.session_state:
    model = train_and_predict(x_train, x_test, x_holdout, img_width, img_height, batch_size, no_of_classes)
    st.session_state.model_trained = True

# Upload an image for prediction
uploaded_file = st.file_uploader("Upload an image of Indian Sign Language", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    if uploaded_file:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        prediction = train_and_predict(x_train, x_test, x_holdout, img_width, img_height, batch_size, no_of_classes, uploaded_file=uploaded_file)
        st.write(f"Predicted Class: {prediction}")
    else:
        st.warning("Please upload an image to predict.")
