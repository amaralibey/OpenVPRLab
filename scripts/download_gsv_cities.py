import sys
import subprocess
import pathlib


def is_kaggle_installed():
    try:
        import kaggle
        return True
    except ImportError:
        return False

def download_dataset(download_path):
    try:
        subprocess.check_call(["kaggle", "datasets", "download",  "amaralibey/gsv-cities", "-p", download_path, "unzip"])
        print(f"GSV-Cities downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

def main():
    if not is_kaggle_installed():
        please = input("Please install Kaggle and try again. Would you like to install Kaggle now? (y/n): ")
        if please.lower()[0] == "y":
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print("\nKaggle installed successfully.")

    # gsv-cities will be downloaded to OpenCPRLav/data/train/gsv-ciies/, click enter to confirm 
    # or enter the path where you want to download the dataset
    from src import _PROJECT_ROOT
    while True:
        path = input(f"Press enter to use default path `{_PROJECT_ROOT}/OpenCPRLav/data/train/gsv-ciies/`, or enter a custom path:")
        if path:
            download_path = pathlib.Path(path)
            # check if the path exists
            if not download_path.exists():
                print(f"Path does not exist: {download_path}")
    
    # check if gsv-cities path is in config file and if the dataset is already downloaded
    # if not, download the dataset

    dataset_name = input("Enter the Kaggle dataset name (e.g., 'username/dataset-name'): ")
    download_dataset(dataset_name)

if __name__ == "__main__":
    main()