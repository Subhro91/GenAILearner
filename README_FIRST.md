# PDF Learning Assistant

This application helps you extract knowledge from PDF documents by generating summaries and interactive quizzes.

## Quick Start Guide

### Windows Users

1. Simply double-click the `setup.bat` file
2. Follow the on-screen instructions
3. When the application is running, open your browser and go to: http://localhost:8080

### macOS/Linux Users

1. Open a terminal in this directory
2. Run the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

3. Open your browser and go to: http://localhost:8080

## Important Notes

### First Run

- The first run will take longer (5-10 minutes) as it downloads the AI models
- The application uses two AI models:
  - `sshleifer/distilbart-cnn-12-6` (1.2GB) for summarization
  - `google/flan-t5-large` (3.1GB) for quiz generation
- These models are downloaded automatically and cached for future use
- You need approximately 5GB of free disk space for the models

### System Requirements

- Python 3.9 or higher
- At least 8GB RAM recommended
- At least 5GB free disk space
- Internet connection for the first run (to download models)

### Using the Application

1. Upload a PDF file using the web interface
2. The application will process the PDF and generate:
   - A summary of the content
   - Interactive quiz questions based on the content
3. You can view both the summary and quiz in the web interface
4. All uploaded PDFs are stored in the `uploaded_pdfs` folder

### Troubleshooting

- If you get "address already in use" errors, try changing the port in the command:
  ```
  uvicorn main:app --host 0.0.0.0 --port 8081 --reload
  ```
- If you have issues with model downloads, check your internet connection
- Make sure your firewall/antivirus is not blocking Python or the downloads

## Advanced Configuration

See the `setup_guide.md` file for more detailed configuration options and information about the application.

## Credits

This application uses:
- FastAPI for the web framework
- PyMuPDF for PDF processing
- Hugging Face Transformers for AI models
- Bootstrap for the web interface 