import streamlit as st
from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import os
import re
import json
import base64
from groq import Groq
from st_keyup import st_keyup
from st_img_pastebutton import paste
from text_highlighter import text_highlighter

if 'cleaned_text' not in st.session_state:
    st.session_state.cleaned_text = ""
if 'polished_text' not in st.session_state:
    st.session_state.polished_text = ""

# Page configuration
st.set_page_config(page_title="DualTextOCRFusion",
                   page_icon="🔍", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load GOT Models


@st.cache_resource
def init_got_model():
    tokenizer = AutoTokenizer.from_pretrained(
        'srimanth-d/GOT_CPU', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'srimanth-d/GOT_CPU', trust_remote_code=True, pad_token_id=tokenizer.eos_token_id)
    return model.eval(), tokenizer


@st.cache_resource
def init_got_gpu_model():
    tokenizer = AutoTokenizer.from_pretrained(
        'ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True,
                                      device_map='cuda', pad_token_id=tokenizer.eos_token_id)
    return model.eval().cuda(), tokenizer

# Load Qwen Model


@st.cache_resource
def init_qwen_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", device_map="cpu", torch_dtype=torch.float16)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model.eval(), processor

# Text Cleaning AI - Clean spaces, handle dual languages


def clean_extracted_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    cleaned_text = re.sub(r'\s([?.!,])', r'\1', cleaned_text)
    return cleaned_text

# Polish the text using a model


def polish_text_with_ai(cleaned_text):
    prompt = f"Remove unwanted spaces between and inside words to join incomplete words, creating a meaningful sentence in either Hindi, English, or Hinglish without altering any words from the given extracted text. Then, return the corrected text with adjusted spaces, keeping it as close to the original as possible, along with relevant details or insights that an AI can provide about the extracted text. Extracted Text: {cleaned_text}"
    client = Groq(
        api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a pedantic sentence corrector. Remove extra spaces between and within words to make the sentence meaningful in English, Hindi, or Hinglish, according to the context of the sentence, without changing any words."
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gemma2-9b-it",
    )
    polished_text = chat_completion.choices[0].message.content
    return polished_text

# Extract text using GOT


def extract_text_got(image_file, model, tokenizer):
    return model.chat(tokenizer, image_file, ocr_type='ocr')

# Extract text using Qwen


def extract_text_qwen(image_file, model, processor):
    try:
        image = Image.open(image_file).convert('RGB')
        conversation = [{"role": "user", "content": [{"type": "image"}, {
            "type": "text", "text": "Extract text from this image."}]}]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[
                           image], return_tensors="pt")
        output_ids = model.generate(**inputs)
        output_text = processor.batch_decode(
            output_ids, skip_special_tokens=True)
        return output_text[0] if output_text else "No text extracted from the image."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to highlight the keyword in the text


def highlight_text(cleaned_text, start, end):
    text_highlighter(
        text=cleaned_text,
        labels=[("KEYWORD", "#0000FF")],
        annotations=[
            {"start": start, "end": end, "tag": "KEYWORD"},
        ],
    )


# Title and UI
st.title("DualTextOCRFusion - 🔍")
st.header("OCR Application - Multimodel Support")
st.write("Upload an image for OCR using various models, with support for English, Hindi, and Hinglish.")

# Sidebar Configuration
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Select OCR Model:", ("GOT_CPU", "GOT_GPU", "Qwen"))

# Upload Section
uploaded_file = st.sidebar.file_uploader(
    "Choose An Image:", type=["png", "jpg", "jpeg"])

# Input from clipboard
# Paste image button
clipboard_use = False
image_data = paste(label="Paste From Clipboard", key="image_clipboard")
if image_data is not None:
    clipboard_use = True
    header, encoded = image_data.split(",", 1)
    decoded_bytes = base64.b64decode(encoded)
    img_stream = io.BytesIO(decoded_bytes)
    uploaded_file = img_stream

# Input from camera
camera_file = st.sidebar.camera_input("Capture From Camera:")
if camera_file:
    uploaded_file = camera_file

# Predict button
predict_button = st.sidebar.button("Predict")

# Main columns
col1, col2 = st.columns([2, 1])

cleaned_text = ""
polished_text = ""

# Display image preview
if uploaded_file:
    image = Image.open(uploaded_file)
    with col1:
        col1.image(image, caption='Uploaded Image',
                   use_column_width=False, width=300)

    # Save uploaded image to 'images' folder
    images_dir = 'images'
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(
        images_dir, "temp_file.png" if clipboard_use else uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Check if the result already exists
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(
        results_dir, "temp_file_result.json" if clipboard_use else f"{uploaded_file.name}_result.json")
    # Handle predictions
    if predict_button:
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            extracted_text = result_data["extracted_text"]
            cleaned_text = result_data["cleaned_text"]
            polished_text = result_data["polished_text"]
        else:
            with st.spinner("Processing..."):
                if model_choice == "GOT_CPU":
                    got_model, tokenizer = init_got_model()
                    extracted_text = extract_text_got(
                        image_path, got_model, tokenizer)

                elif model_choice == "GOT_GPU":
                    got_gpu_model, tokenizer = init_got_gpu_model()
                    extracted_text = extract_text_got(
                        image_path, got_gpu_model, tokenizer)

                elif model_choice == "Qwen":
                    qwen_model, qwen_processor = init_qwen_model()
                    extracted_text = extract_text_qwen(
                        image_path, qwen_model, qwen_processor)

                cleaned_text = clean_extracted_text(extracted_text)
                polished_text = polish_text_with_ai(cleaned_text) if model_choice in [
                    "GOT_CPU", "GOT_GPU"] else cleaned_text

                # Save results to JSON file
                result_data = {"extracted_text": extracted_text,
                               "cleaned_text": cleaned_text, "polished_text": polished_text}
                with open(result_path, 'w') as f:
                    json.dump(result_data, f)

        # Save results to session state
        st.session_state.cleaned_text = cleaned_text
        st.session_state.polished_text = polished_text

# Display extracted text
st.subheader("Extracted Text (Cleaned & Polished)")
if st.session_state.cleaned_text:
    st.markdown(st.session_state.cleaned_text, unsafe_allow_html=True)
if st.session_state.polished_text:
    st.markdown(st.session_state.polished_text, unsafe_allow_html=True)

# Input search term
search_term = st.text_input("Search Keywords (Update live):")

# Highlight search results in real-time
if search_term and st.session_state.cleaned_text:
    search_keywords = search_term.split()
    for keyword in search_keywords:
        # Find all matches of the keyword in the text and apply highlighting
        matches = re.finditer(re.escape(keyword),
                              st.session_state.cleaned_text, re.IGNORECASE)
        for match in matches:
            start, end = match.span()
            highlight_text(st.session_state.cleaned_text, start, end)

    # Display the highlighted text in the output section
    col2.subheader("Highlighted Text with Keywords")
    highlighted_text = text_highlighter(
        text=st.session_state.cleaned_text,
        labels=[("KEYWORD", "#ffcc00")],  # Color for the highlight
        annotations=[
            {"start": match.start(), "end": match.end(), "tag": "KEYWORD"}
            for keyword in search_keywords
            for match in re.finditer(re.escape(keyword), st.session_state.cleaned_text, re.IGNORECASE)
        ],
    )
    col2.write(highlighted_text, unsafe_allow_html=True)
