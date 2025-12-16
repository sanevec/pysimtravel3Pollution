import os
import re

base_path = "simulationData32Last"

# Regex pattern to find np.float32(number) and replace it with just number
pattern = re.compile(r'np\.float32\((.*?)\)')

for filename in os.listdir(base_path):
    # Check if the file name contains np.float32(number)
    if 'np.float32' in filename:
        # Replace np.float32(number) with number
        new_filename = pattern.sub(r'\1', filename)
        # Construct the full old and new file paths
        old_filepath = os.path.join(base_path, filename)
        new_filepath = os.path.join(base_path, new_filename)
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed {old_filepath} to {new_filepath}")
