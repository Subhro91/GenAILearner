import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import sys
import time

def download_models():
    print("=" * 50)
    print("Downloading AI models - this may take 5-10 minutes on first run")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {device}")
    
    # Add debug information about cache locations
    print(f"Hugging Face cache location (HF_HOME): {os.environ.get('HF_HOME', 'Not set')}")
    print(f"Transformers cache location (TRANSFORMERS_CACHE): {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
    print(f"Default cache location: {os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        # Download summarization model
        print("\nDownloading summarization model (distilbart-cnn-12-6)...")
        start_time = time.time()
        summarizer = pipeline("summarization", 
                             model="sshleifer/distilbart-cnn-12-6", 
                             max_length=65,
                             min_length=30,
                             device=device_id)
        download_time = time.time() - start_time
        print(f"✓ Summarization model downloaded in {download_time:.2f} seconds")
        
        # Download QA model
        print("\nDownloading question generation model (flan-t5-large)...")
        start_time = time.time()
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        download_time = time.time() - start_time
        print(f"✓ Question generation model downloaded in {download_time:.2f} seconds")
        
        # Save models locally
        print("\nSaving models locally for faster loading...")
        start_time = time.time()
        summarizer.save_pretrained(os.path.join("models", "summarizer"))
        tokenizer.save_pretrained(os.path.join("models", "flan-t5-tokenizer"))
        model.save_pretrained(os.path.join("models", "flan-t5-model"))
        save_time = time.time() - start_time
        print(f"✓ Models saved locally in {save_time:.2f} seconds")
        
        # Print model sizes
        summarizer_size = get_directory_size(os.path.join("models", "summarizer"))
        tokenizer_size = get_directory_size(os.path.join("models", "flan-t5-tokenizer"))
        model_size = get_directory_size(os.path.join("models", "flan-t5-model"))
        total_size = summarizer_size + tokenizer_size + model_size
        print(f"\nModel sizes:")
        print(f"- Summarizer: {summarizer_size / (1024*1024):.2f} MB")
        print(f"- Tokenizer: {tokenizer_size / (1024*1024):.2f} MB")
        print(f"- Question generation model: {model_size / (1024*1024):.2f} MB")
        print(f"- Total: {total_size / (1024*1024):.2f} MB")
        
        print("\n✓ All models downloaded and saved successfully!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\nERROR: Failed to download models: {str(e)}")
        print("\nPlease check your internet connection and try again.")
        print("If the problem persists, you may need to manually download the models.")
        print("=" * 50)
        return False

def get_directory_size(path):
    """Get the size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1) 