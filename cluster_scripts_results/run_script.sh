#!/bin/sh

# Create the OK_Measurements directory
mkdir -p OK_Measurements tmp_unzip_dir

apt-get update && apt-get install -y python3 python3-venv python3-pip unzip


# Unzip the data/Test_202402-4.zip file into the temporary directory
unzip OK_Measurements.zip -d tmp_unzip_dir

# Move the contents of the temporary directory to OK_Measurements
mv tmp_unzip_dir/*/* OK_Measurements/

# Remove the temporary directory
rm -rf tmp_unzip_dir

# Print the current working directory
echo "Current working directory: $(pwd)"

# List the contents of the current directory and its subdirectories
echo "Contents of current directory:"
ls -R

# Create a virtual environment in the home directory
python3 -m venv /tmp/venv

# Activate the virtual environment
. /tmp/venv/bin/activate


# Upgrade pip
pip install --upgrade pip

# Install necessary Python packages
pip install matplotlib tensorflow pandas scikit-learn pyarrow numpy imblearn

# Run the main script
python3 DataScience/src/main.py


