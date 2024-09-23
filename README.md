# 🔍 DualTextOCRFusion

**DualTextOCRFusion** is a web-based Optical Character Recognition (OCR) application that allows users to upload images containing both Hindi and English text, extract the text, and search for keywords within the extracted text. The app uses advanced models like **ColPali’s Byaldi + Qwen2-VL** or **General OCR Theory (GOT)** for multilingual text extraction.

## Features

- **Multilingual OCR**: Extract text from images containing both **Hindi** and **English**.
- **Keyword Search**: Search for specific keywords in the extracted text.
- **User-Friendly Interface**: Simple, intuitive interface for easy image uploading and searching.
- **Deployed Online**: Accessible through a live URL for easy use.

## Technologies Used

- **Python**: Backend logic.
- **Streamlit**: For building the web interface.
- **Huggingface Transformers**: For integrating OCR models (Qwen2-VL or GOT).
- **PyTorch**: For deep learning inference.
- **Pytesseract**: Optional OCR engine.
- **OpenCV**: For image preprocessing.

## Project Structure

```
DualTextOCRFusion/
│
├── app.py                 # Main Streamlit application
├── ocr.py                 # Handles OCR extraction using the selected model
├── .gitignore             # Files and directories to ignore in Git
├── .streamlit/
│   └── config.toml        # Streamlit theme configuration
├── requirements.txt       # Dependencies for the project
└── README.md              # This file
```

## How to Run Locally

### Prerequisites

- Python 3.8 or above installed on your machine.
- Tesseract installed for using `pytesseract` (optional if using Huggingface models). You can download Tesseract from [here](https://github.com/tesseract-ocr/tesseract).

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/dual-text-ocr-fusion.git
   cd dual-text-ocr-fusion
   ```

2. **Install Dependencies**:

   Make sure you have the required dependencies by running the following:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   Start the Streamlit app by running the following command:

   ```bash
   streamlit run app.py
   ```

4. **Open the App**:

   Once the server starts, the app will be available in your browser at:

   ```
   http://localhost:8501
   ```

### Usage

1. **Upload an Image**: Upload an image containing Hindi and English text in formats like JPG, JPEG, or PNG.
2. **View Extracted Text**: The app will extract and display the text from the image.
3. **Search for Keywords**: Enter any keyword to search within the extracted text.

## Deployment

The app is deployed on **Streamlit Sharing** and can be accessed via the live URL:

**[Live Application](https://your-app-link.streamlit.app)**

## Customization

### Changing the OCR Model

By default, the app uses the **Qwen2-VL** model, but you can switch to the **General OCR Theory (GOT)** model by editing the `ocr.py` file.

- **For Qwen2-VL**:
  
  ```python
  from ocr import extract_text_byaldi
  ```

- **For General OCR Theory (GOT)**:
  
  ```python
  from ocr import extract_text_got
  ```

### Custom UI Theme

You can customize the look and feel of the application by modifying the `.streamlit/config.toml` file. Adjust colors, fonts, and layout options to suit your preferences.

## Example Images

Here are some sample images you can use to test the OCR functionality:

1. **Sample 1**: A document with mixed Hindi and English text.
2. **Sample 2**: An image with only Hindi text for multilingual OCR testing.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Follow these steps:

1. Fork the project.
2. Create a feature branch:

   ```bash
   git checkout -b feature-branch
   ```

3. Commit your changes:

   ```bash
   git commit -am 'Add new feature'
   ```

4. Push to the branch:

   ```bash
   git push origin feature-branch
   ```

5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **Streamlit**: For the easy-to-use web interface.
- **Huggingface Transformers**: For the powerful OCR models.
- **Tesseract**: For optional OCR functionality.
- **ColPali & GOT Models**: For the multilingual OCR support.