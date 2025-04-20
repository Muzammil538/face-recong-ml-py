import os
import requests


def download_file(url, file_path):
    """
    Download a file using requests library to avoid SSL issues.

    Args:
        url: URL of the file to download
        file_path: Path where the file should be saved
    """
    try:
        print(f"Downloading {url} to {file_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {file_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False