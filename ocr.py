import os
from transformers import AutoModel, AutoTokenizer
import torch
from torch.cuda.amp import autocast

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16  # Use float16 on CUDA
    print(f"CUDA device available, using {device} with {dtype}.")
else:
    device = torch.device("cpu")
    dtype = torch.float32  # Use float32 on CPU
    print(f"CUDA not available, using {device} with {dtype}.")

# Load model and tokenizer
model_name = 'ucaslcl/GOT-OCR2_0'
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True, return_tensors='pt')

# Load the model
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
)

# Convert model to FP16 for inference
if device.type == 'cuda':
    model = model.half()  # Converts model to float16

model = model.eval().to(device).float()


def extract_text_got(uploaded_file):
    """Use GOT-OCR2.0 model to extract text from the uploaded image."""
    try:
        temp_file_path = 'temp_image.jpg'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())  # Save file

        # OCR attempts
        ocr_types = ['ocr', 'format']
        fine_grained_options = ['ocr', 'format']
        color_options = ['red', 'green', 'blue']
        box = [10, 10, 100, 100]  # Example box for demonstration
        multi_crop_types = ['ocr', 'format']

        results = []

        # Apply autocast for mixed precision inference
        with autocast():
            # Try PLAIN and FORMATTED OCR
            for ocr_type in ocr_types:
                with torch.no_grad():
                    outputs = model.chat(
                        tokenizer, temp_file_path, ocr_type=ocr_type)
                    if isinstance(outputs, list) and outputs[0].strip():
                        return outputs[0].strip()  # Return if successful
                    results.append(outputs[0].strip()
                                   if outputs else "No result")

            # Try FINE-GRAINED OCR with box options
            for ocr_type in fine_grained_options:
                with torch.no_grad():
                    outputs = model.chat(
                        tokenizer, temp_file_path, ocr_type=ocr_type, ocr_box=box)
                    if isinstance(outputs, list) and outputs[0].strip():
                        return outputs[0].strip()  # Return if successful
                    results.append(outputs[0].strip()
                                   if outputs else "No result")

            # Try FINE-GRAINED OCR with color options
            for ocr_type in fine_grained_options:
                for color in color_options:
                    with torch.no_grad():
                        outputs = model.chat(
                            tokenizer, temp_file_path, ocr_type=ocr_type, ocr_color=color)
                        if isinstance(outputs, list) and outputs[0].strip():
                            return outputs[0].strip()  # Return if successful
                        results.append(outputs[0].strip()
                                       if outputs else "No result")

            # Try MULTI-CROP OCR
            for ocr_type in multi_crop_types:
                with torch.no_grad():
                    outputs = model.chat_crop(
                        tokenizer, temp_file_path, ocr_type=ocr_type)
                    if isinstance(outputs, list) and outputs[0].strip():
                        return outputs[0].strip()  # Return if successful
                    results.append(outputs[0].strip()
                                   if outputs else "No result")

        # If no text was extracted
        if all(not text for text in results):
            return "No text extracted."
        else:
            return "\n".join(results)

    except Exception as e:
        return f"Error during text extraction: {str(e)}"

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
