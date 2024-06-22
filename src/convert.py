import os
import pandas as pd
from tqdm import tqdm
from .utils import list_files_and_directories
from time import time
from multiprocessing import Pool, cpu_count
from colorama import Fore
import shutil

def convert_single_file(arguments):
    """Convert a single file.

    Args:
        arguments (tuple[2]): packed arguments as follows:
            args (argparse.Namespace): CLI arguments
            rel_filepath (str): Relative file path
    """
    args, rel_filepath = arguments
    
    real_filepath = os.path.join(args.input_dir, rel_filepath)
    f = open(real_filepath, "r")
    f.readline()
    header_org = f.readline().strip().split(",")
    f.close()
    
    df = pd.read_csv(real_filepath, skiprows=4, header=None)
    
    convert_header = ["Frequency (Hz)"]
    convert_values = [df[0]]
    record_color = ()
    prev_color = ""
    for i in range(0, len(header_org)-1, 3):
        color, _, _, axis = header_org[i].split(" ")
        if prev_color != color:
            record_color += (prev_color,)
        if color not in record_color:
            convert_header.append(f"{color}_{axis}")
        else:
            convert_header.append(f"{color}2_{axis}")
            
        prev_color = color
        convert_values.append(df[i+1])
        
    odf = pd.DataFrame(dict(zip(convert_header, convert_values)))
    save_path = os.path.join(args.output_dir, rel_filepath)
    odf.to_csv(save_path, index=False)

def convert_files_in_parallel(prompt, args, files):
    """Convert files in parallel.

    Args:
        prompt (str): Prompt message
        args (argparse.Namespace): CLI arguments
        files (list): List of files
        output_dir (str): Output directory

    Returns:
        list: List of results
    """

    num_cores = max(1, cpu_count()//2) if args.ncpu == -1 else args.ncpu
    arguments = [(args, rel_filepath) for rel_filepath in files]

    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(convert_single_file, arguments), total=len(files), desc=prompt, leave=True))

    return results

def convert(args):
    """Convert ugly original spectrum format into better presentation for further analysis.

    Args:
        args (argparse.Namespace): CLI arguments
    """
    start_time = time()
    
    # Create output directory and neccesary subdirectories if they does not exist
    if (os.path.exists(args.output_dir) == False):
        os.makedirs(args.output_dir)
    else:
        if args.force == 1:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            print(Fore.RED + f"Output directory {args.output_dir} already exists. Please remove it or use -f option to overwrite.")
            return
    files, dirs = list_files_and_directories(args.input_dir)
    for d in dirs:
        if (os.path.exists(os.path.join(args.output_dir, d)) == False):
            os.makedirs(os.path.join(args.output_dir, d))
    
    # Convert files
    convert_files_in_parallel("Converting files...", args, files)
    
    print(f"Conversion completed in {time() - start_time:.2f} seconds.")