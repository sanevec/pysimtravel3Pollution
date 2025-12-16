import os
import numpy as np


# Define the directories
folder1 = "simulationData"  # Replace with the correct path to your first folder
folder2 = "simulationData32"  # Replace with the correct path to your second folder
substring = "densityCars_0.15"


# List all files in both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# Find common filenames that start with 'P_'
common_files = {filename for filename in files1.intersection(files2) if filename.startswith('P_') and substring in filename}
print(len(common_files))

if common_files:
    # Select the first common file
    filename = next(iter(common_files))
    file_path1 = os.path.join(folder1, filename)
    file_path2 = os.path.join(folder2, filename)
    
    data1 = np.load(file_path1)
    data2 = np.load(file_path2)
    
    print(f"Loaded {filename} from both folders")
    #print("Keys in data1:", data1.files)
    #print("Keys in data2:", data2.files)



    # Example processing of loaded data:
    # You can access arrays stored in the npz file like this:
    array1 = data1["1_CO2"][:,:,:-1]
    array2 = data2["1_CO2"][:,:,:-1]
    print((array1==array2).all())
    #print(array2)

    print((np.max(np.abs(array1-array2))))
    print((np.max(np.abs(array1))))
    print((np.max(np.abs(array2))))
    
    '''
    suma_array1 = np.sum(array1, axis=(0, 1))
    suma_array2 = np.sum(array2, axis=(0, 1))

    #print((array1-array2==np.zeros_like(array2)).all())
    print(suma_array1)
    print(suma_array2)


    if np.array_equal(array1, array2):
        print("The arrays are exactly the same.")
    else:
        print("The arrays are different.")

    # Define a tolerance level
    tolerance = 1e-1
    
    if np.allclose(array1, array2, atol=tolerance):
        print("The arrays are similar within the given tolerance.")
    else:
        print("The arrays are different.")

    
    # 3. Compute the Difference and Check Statistics
    difference = array1 - array2
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)

    print(f"Mean difference: {mean_diff}")
    print(f"Standard deviation of difference: {std_diff}")

    mean_threshold = 1e-5
    std_threshold = 1e-5
    if abs(mean_diff) < mean_threshold and std_diff < std_threshold:
        print("The arrays are similar within the defined thresholds.")
    else:
        print("The arrays are different.")


    relative_error = np.abs(array1 - array2) / (np.abs(array1) + np.abs(array2))

    # Handle division by zero if both elements are zero
    relative_error = np.where((array1 == 0) & (array2 == 0), 0, relative_error)

    # Print the relative error
    print("Relative error between the arrays:")
    #print(relative_error)

    # Optionally, you can set a threshold to determine if they are similar
    threshold = 1e-5
    if np.all(relative_error < threshold):
        print("The arrays are similar within the given relative error threshold.")
    else:
        print("The arrays are different.")

else:
    print("No common files starting with 'P_' found in both folders.")
'''