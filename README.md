# PDF Learning Assistant

A powerful tool that uses AI to help you learn from PDF documents. Upload your PDFs, get summaries, and generate quizzes to test your knowledge.

## Features

- **PDF Upload**: Upload any PDF document for analysis
- **Text Extraction**: Extract and process text from PDFs
- **AI-Powered Summarization**: Get concise summaries of your documents
- **Quiz Generation**: Create quizzes based on the content to test your understanding
- **User-Friendly Interface**: Simple and intuitive web interface

## System Requirements

- Windows 10 or higher
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space
- Internet connection (for initial setup and model download)

## Installation and Setup

### First-Time Setup

1. Download and extract the application files to a location on your computer
2. Double-click `setup.bat` to:
   - Create a Python virtual environment
   - Install all required dependencies
   - Download the AI models (this will take 5-10 minutes on first run)
   - Start the application

### Running the Application After Setup

Once you've completed the initial setup, you can use:

- `run_app.bat` to start the application without repeating the setup process

### Downloading Models Only

If you need to download or update the AI models without running the application:

- Run `download_models.bat` to download just the AI models

## Usage

1. Start the application using `setup.bat` (first time) or `run_app.bat` (subsequent times)
2. Open your web browser and go to: http://localhost:8080
3. Upload a PDF document using the upload button
4. Wait for the document to be processed
5. View the summary and generate quizzes as needed

## Troubleshooting

### Models Not Downloading

If you encounter issues with the AI models not downloading:

1. Make sure you have a stable internet connection
2. Try running `download_models.bat` separately
3. Check that you have enough disk space (at least 2GB free)

### Application Not Starting

If the application fails to start:

1. Make sure Python is installed and added to your PATH
2. Try running `setup.bat` again to reinstall dependencies
3. Check the console for specific error messages

## Technical Details

This application uses:

- FastAPI for the web server
- PyMuPDF for PDF processing
- Hugging Face Transformers for AI models:
  - DistilBART for summarization
  - FLAN-T5 for quiz generation

## License

This software is provided for educational purposes only. See the LICENSE file for details. 