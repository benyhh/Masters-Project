import os

folder1 = "./PointingScanPlots_v2/Bad/"
folder2 = "./PointingScanPlots_v2/Good/"

# Get a list of all files in folder1
files1 = os.listdir(folder1)

# Get a list of all files in folder2
files2 = os.listdir(folder2)

# Find the files that are in both folder1 and folder2
files_to_delete = list(set(files1) & set(files2))

print(files_to_delete)
# Delete the files in folder2
"""for file in files_to_delete:
    file_path = os.path.join(folder2, file)
    os.remove(file_path)"""
