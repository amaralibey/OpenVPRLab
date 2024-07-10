#!/bin/bash


# Get the project root directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"


# # Set the dataset name and download path
# DATASET_NAME="msls-val"
DOWNLOAD_URL="https://github.com/amaralibey/OpenVPRLab/releases/download/v0.1/msls-val.zip"
DOWNLOAD_PATH="$PROJECT_ROOT/data/val/msls-val"
# ZIP_FILE="$DOWNLOAD_PATH/$DATASET_NAME.zip"

# # Create the download directory
mkdir -p "$DOWNLOAD_PATH" || error "Failed to create directory: $DOWNLOAD_PATH"


# # Check if the dataset has already been downloaded
# if [ -d "$DOWNLOAD_PATH" ] && [ "$(ls -A "$DOWNLOAD_PATH")" ]; then
#     echo "The folder $DOWNLOAD_PATH is not empty, have you already downloaded the dataset? Please check $DOWNLOAD_PATH"
#     exit 0
# fi

# Download the dataset
echo "Downloading $DATASET_NAME dataset for OpenVPRLab..."
echo "Saving to path: $DOWNLOAD_PATH/msls-val.zip"

if ! curl -L -o "$DOWNLOAD_PATH/msls-val.zip" "$DOWNLOAD_URL"; then
    error "Failed to download the dataset."
fi


# # Unzip the file
echo "Download successful. Unzipping the file..."
if ! unzip -q "$DOWNLOAD_PATH/msls-val.zip" -d "$DOWNLOAD_PATH"; then
    error "Failed to unzip the file."
fi

# Remove the zip file after successful extraction
rm "$DOWNLOAD_PATH/msls-val.zip" || echo "Warning: Failed to remove zip file: $DOWNLOAD_PATH/msls-val.zip"

echo "Dataset successfully downloaded and extracted to $DOWNLOAD_PATH"