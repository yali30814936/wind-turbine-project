import os

def get_freq_delta(filepath):
    """Calculate the frequency delta from the given file, which will be used to calculate the smooth window size.

    Args:
        filepath (str): File path

    Returns:
        float: Frequency delta
    """
    fp = open(filepath, 'r')
    fp.readline()
    freq1 = float(fp.readline().strip().split(',')[0])
    freq2 = float(fp.readline().strip().split(',')[0])
    return freq2 - freq1

def list_files_and_directories(directory):
    """Lists all files and directories within a given directory.

    Args:
        directory (str): The root directory to walk through.

    Returns:
        tuple: A tuple containing two lists:
            file_list (list): Relative paths of all files found.
            dir_list  (list): Relative paths of all directories found.
    """
    file_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            relative_file_path = os.path.relpath(os.path.join(root, file), directory)
            file_list.append(relative_file_path)
        for dir in dirs:
            relative_dir_path = os.path.relpath(os.path.join(root, dir), directory)
            dir_list.append(relative_dir_path)
    return file_list, dir_list