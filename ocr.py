import os
from transformers import AutoModel, AutoTokenizer
import torch

# Load model, tokenizer, and processor once to improve performance
model_name = 'stepfun-ai/GOT-OCR2_0'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and map to the appropriate device (GPU or CPU)
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True, low_cpu_mem_usage=True, use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model.to(device)


def extract_text_got(uploaded_file):
    """Use GOT-OCR2.0 model to extract text from the uploaded image."""
    try:
        # Save the uploaded file temporarily
        temp_file_path = 'temp_image.jpg'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())  # Save the file

        # Define different OCR attempts
        ocr_types = ['ocr', 'format']
        fine_grained_options = ['ocr', 'format']
        color_options = ['red', 'green', 'blue']
        box = [10, 10, 100, 100]  # Example box for demonstration
        multi_crop_types = ['ocr', 'format']

        # Store results for each attempt
        results = []

        # Try PLAIN and FORMATTED OCR
        for ocr_type in ocr_types:
            with torch.no_grad():
                outputs = model.chat(
                    tokenizer, temp_file_path, ocr_type=ocr_type)
                extracted_text = outputs[0]  # Assuming outputs is a list
            if extracted_text.strip():
                return extracted_text.strip()  # Return if successful
            results.append(extracted_text.strip())

        # Try FINE-GRAINED OCR (with box options)
        for ocr_type in fine_grained_options:
            with torch.no_grad():
                outputs = model.chat(
                    tokenizer, temp_file_path, ocr_type=ocr_type, ocr_box=box)
                extracted_text = outputs[0]
            if extracted_text.strip():
                return extracted_text.strip()
            results.append(extracted_text.strip())

        # Try FINE-GRAINED OCR (with color options)
        for ocr_type in fine_grained_options:
            for color in color_options:
                with torch.no_grad():
                    outputs = model.chat(
                        tokenizer, temp_file_path, ocr_type=ocr_type, ocr_color=color)
                    extracted_text = outputs[0]
                if extracted_text.strip():
                    return extracted_text.strip()
                results.append(extracted_text.strip())

        # Try MULTI-CROP OCR
        for ocr_type in multi_crop_types:
            with torch.no_grad():
                outputs = model.chat_crop(
                    tokenizer, temp_file_path, ocr_type=ocr_type)
                extracted_text = outputs[0]
            if extracted_text.strip():
                return extracted_text.strip()
            results.append(extracted_text.strip())

        # If no text was successfully extracted, return results
        if all(not text for text in results):
            return "No text extracted."
        else:
            return results

    except Exception as e:
        return f"Error during text extraction: {str(e)}"

    finally:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
