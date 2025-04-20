import os
from pathlib import Path
from utils.downloader import download_file


def setup_models():
    """
    Download and setup required model files if they don't exist.

    Returns:
        bool: True if all downloads were successful, False otherwise
    """
    # Create models directory if it doesn't exist
    model_dir = os.path.abspath("tf-pose-estimation/models/graph/mobilenet_thin_432x368/")
    os.makedirs("models", exist_ok=True)

    # Download pose detection model files if they don't exist
    model_url = "https://github.com/ZheC/tf-pose-estimation/blob/master/models/graph/mobilenet_thin_432x368/graph_opt.pb"
    model_path = os.path.join(model_dir, "graph_opt.pb")

    if not os.path.exists(model_path):
        print(f"Downloading TensorFlow pose model to {model_path}...")
        return download_file(model_url, model_path)

    print("Model already exists.")
    return True


def create_sample_directory():
    """
    Creates a sample directory structure for known faces if none exists.
    """
    sample_dir = "image_data"
    os.makedirs(sample_dir, exist_ok=True)

    # Check if directory is empty
    if not any(Path(sample_dir).iterdir()):
        print("Created sample directory structure at 'known_faces/sample_user'")
        print("Please add face images to this directory before running face recognition.")