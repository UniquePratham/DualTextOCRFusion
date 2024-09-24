import streamlit as st
from ocr import extract_text_got  # The updated OCR function
import json
import numpy

# --- UI Styling ---
st.set_page_config(page_title="DualTextOCRFusion",
                   layout="centered", page_icon="🔍")

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f4f4f4;
    }
    .sidebar .sidebar-content {
        background: #e0e0e0;
    }
    h1 {
        color: #007BFF;
    }
    .upload-btn {
        background-color: #007BFF;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Title ---
st.title("🔍 DualTextOCRFusion")
st.write("Upload an image with **Hindi** and **English** text to extract and search for keywords.")

# --- Image Upload Section ---
uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Extract text from the image using the selected OCR function (GOT)
    with st.spinner("Extracting text using the model..."):
        try:
            extracted_text = extract_text_got(
                uploaded_file)  # Pass uploaded_file directly
            if not extracted_text.strip():
                st.warning("No text extracted from the image.")
        except Exception as e:
            st.error(f"Error during text extraction: {str(e)}")
            extracted_text = ""

    # Display extracted text
    st.subheader("Extracted Text")
    st.text_area("Text", extracted_text, height=250)

    # Save extracted text for search
    if extracted_text:
        with open("extracted_text.json", "w") as json_file:
            json.dump({"text": extracted_text}, json_file)

        # --- Keyword Search ---
        st.subheader("Search for Keywords")
        keyword = st.text_input(
            "Enter a keyword to search in the extracted text")

        if keyword:
            if keyword.lower() in extracted_text.lower():
                st.success(f"Keyword **'{keyword}'** found in the text!")
            else:
                st.error(f"Keyword **'{keyword}'** not found.")
