from huggingface_hub import snapshot_download
import os

def main():
    # Prompt the user for the repo URL
    repo_id = input("Please enter the Hugging Face repo URL (e.g., 'username/repo_name'): ")
    
    local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", repo_id.split('/')[-1], "1")
    print(f"Downloading model to: {local_dir}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir
        )
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

if __name__ == "__main__":
    main()
