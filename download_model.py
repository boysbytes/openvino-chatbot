from huggingface_hub import snapshot_download
import os
import re

def extract_repo_id(url):
    """
    Extracts the repo ID (e.g., 'username/repo_name') from a Hugging Face model URL.
    """
    match = re.match(r'https://huggingface.co/([^/]+/[^/]+)', url)
    if match:
        return match.group(1)  # Extracts "username/repo_name"
    else:
        return None

def main():
    # Prompt the user for the full Hugging Face model URL
    model_url = input("Please enter the full Hugging Face model URL: ").strip()

    # Extract the repo ID from the URL
    repo_id = extract_repo_id(model_url)
    
    if not repo_id:
        print("Invalid Hugging Face URL. Please enter a valid URL like 'https://huggingface.co/username/repo_name'.")
        return

    # Define the local directory for model storage
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", repo_id.split('/')[-1], "1")

    # Check if the model already exists to avoid re-downloading
    if os.path.exists(local_dir):
        print(f"Model already exists at {local_dir}, skipping download.")
        return

    print(f"Downloading model from {model_url} to: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            resume_download=True  # Prevents unnecessary re-downloads
        )
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
