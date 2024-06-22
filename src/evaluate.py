import pandas as pd
import os
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing
from time import time
from colorama import Fore
import shutil

from .utils import get_freq_delta

header = ["Green_X", "Blue_X", "Red_X", "Purple_X", "Yellow_X", "Orange_X", "Green2_X", "Blue2_X", "Red2_X", "Pink2_X", "Yellow2_X", "Orange2_X"]

# Define the grid size
rows = 4
cols = 3

# Generate indices for the grid
x, y = np.meshgrid(np.arange(cols), np.arange(rows))
coords = np.stack((y, x), axis=-1).reshape(-1, 2)

pattern_subdir = 'pattern'
normal_subdir  = 'normal'
abnormal_subdir = 'abnormal'

def plot_pattern(pattern_df_smooth, savdir):
    """plot the normal pattern

    Args:
        pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
        savdir (str): directory to save the plot
    """
    _, axs = plt.subplots(4, 3, figsize=(30, 20), constrained_layout=True)
    ymin = 0
    ymax = 150
    
    colorA = "black"
    alphaA = 1
    
    tick_size = 32
    xaxs = pattern_df_smooth["Frequency (Hz)"]
    
    for i in range(12):
        ax = axs[coords[i][0], coords[i][1]]
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(xaxs, pattern_df_smooth[header[i]], alpha=alphaA, color=colorA)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, 650)
        ax.set_xticks([100, 300, 500, 650])
        
    plt.savefig(os.path.join(savdir, "pattern.png"))
    plt.close()

def predict(file_info):
    """perform prediction on a single file

    Args:
        file_info (tupe[5]): packed arguments as follows:
            args (argparse.Namespace): CLI arguments
            filename (str): file path
            savdir (str): directory to save the plot
            pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
            smooth_windows_len (int): window size for smoothing

    Returns:
        int: the prediction result, 1 for abnormal, 0 for normal
    """
    args, filename, savdir, pattern_df_smooth, smooth_windows_len = file_info
    
    df = pd.read_csv(filename)[["Frequency (Hz)"] + header]
    df_smooth = df.rolling(window=smooth_windows_len, center=True).mean().dropna()

    # Calculate the mean absolute error 
    # TODO: Change it to any other metric you want
    errors = metrics.mean_absolute_error(pattern_df_smooth.iloc[:, 1:], df_smooth.iloc[:, 1:], multioutput='raw_values')
    # errors = (df_smooth.iloc[:, 1:] - pattern_df_smooth.iloc[:, 1:]).mean().values  # Mean error

    # Initialize the plot
    _, axs = plt.subplots(4, 3, figsize=(30, 20), constrained_layout=True)
    ymin = 0
    ymax = 150
    
    colorA = "black"
    alphaA = 1
    alphaB = 1
    
    tick_size = 32
    xaxs = df_smooth["Frequency (Hz)"]
    
    # Draw each RoI
    if not args.quite:
        for i in range(12):
            # Determine the color of this subplot
            plot_color = "red" if (errors[i] >= args.theta) else "green"

            ax = axs[coords[i][0], coords[i][1]]
            ax.tick_params(axis='both', which='major', labelsize=tick_size)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.plot(xaxs, pattern_df_smooth[header[i]], alpha=alphaA, color=colorA)
            ax.plot(xaxs, df_smooth[header[i]], alpha=alphaB, color=plot_color)
            ax.set_ylim(ymin, ymax)
            ax.set_xlim(0, 650)
            ax.set_xticks([100, 300, 500, 650])
            ax.set_title(f"error={errors[i]:.2f}", fontsize=tick_size, color=plot_color)
            
        plt.savefig(os.path.join(savdir, os.path.basename(filename)[:-4]+".png"))
        plt.close()
    
    if (sum(errors >= args.theta) >= args.threshold_cnt):
        return 1
    else:
        return 0
      

def predict_files_in_parallel(prompt, args, filelist, savdir, pattern_df_smooth, smooth_windows_len):
    """predict files in parallel

    Args:
        prompt (str): prompt message for progress bar
        args (argparse.Namespace): CLI arguments
        filelist (list[str]): list of file paths to be predicted
        savdir (str): directory to save the plot
        pattern_df_smooth (pd.DataFrame): pattern data frame to be plotted
        smooth_windows_len (int): window size for smoothing

    Returns:
        list: list of prediction results
    """
    file_info_list = [(args, filename, savdir, pattern_df_smooth, smooth_windows_len) for filename in filelist]

    num_cores = max(1, multiprocessing.cpu_count()//2) if args.ncpu == -1 else args.ncpu
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(predict, file_info_list), total=len(file_info_list), desc=prompt, leave=True))

    return results

def evaluate(args):
    """Evaluate the performance of the abonomaly detection algorithm.

    Args:
        args (argparse.Namespace): CLI arguments

    Raises:
        ValueError: No pattern files found.
        ValueError: No normal files found.
        ValueError: No abnormal files found.
    """
    start_time = time()
    
    pattern_files  = glob(os.path.join(args.input_dir, pattern_subdir,  '*.csv'))
    normal_files   = glob(os.path.join(args.input_dir, normal_subdir,   '*.csv'))
    abnormal_files = glob(os.path.join(args.input_dir, abnormal_subdir, '*.csv'))
    
    # Check if files exist
    if len(pattern_files) == 0:
        raise ValueError('No pattern files found.')
    if len(normal_files) == 0:
        raise ValueError('No normal files found.')
    if len(abnormal_files) == 0:
        raise ValueError('No abnormal files found.')
    
    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if args.force == 1:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            print(Fore.RED + f"Output directory {args.output_dir} already exists. Please remove it or use -f option to overwrite.")
            return
    
    # Get the frequency delta
    smooth_windows_len = int(args.smooth_window_Hz / get_freq_delta(pattern_files[0]))
    
    # Build the pattern
    pattern_df = None
    for file in tqdm(pattern_files, desc="Calculate the normal pattern...", leave=True):
        df = pd.read_csv(file)[["Frequency (Hz)"] + header]
        if pattern_df is None:
            pattern_df = df
        else:
            pattern_df += df
    pattern_df /= len(pattern_files)
    
    # Smooth the pattern
    pattern_df_smooth = pattern_df.rolling(window=smooth_windows_len, center=True).mean().dropna()
    
    # Draw the pattern
    plot_pattern(pattern_df_smooth, args.output_dir)
    
    # Predict the normal files
    predictions = []
    labels = [0] * len(normal_files)
    normal_savdir = os.path.join(args.output_dir, normal_subdir)
    if (not args.quite) and (not os.path.exists(normal_savdir)):
        os.makedirs(normal_savdir)
    predictions = predict_files_in_parallel("Predicting normal cases...    ", args, normal_files, normal_savdir, pattern_df_smooth, smooth_windows_len)

    # Predict the abnormal files
    labels += [1] * len(abnormal_files)
    abnormal_savdir = os.path.join(args.output_dir, abnormal_subdir)
    if (not args.quite) and (not os.path.exists(abnormal_savdir)):
        os.makedirs(abnormal_savdir)
    predictions += predict_files_in_parallel("Predicting abnormal cases...  ", args, abnormal_files, abnormal_savdir, pattern_df_smooth, smooth_windows_len)
    
    # Save the results
    df = pd.DataFrame({'label': labels, 'prediction': predictions})
    df.to_csv(os.path.join(args.output_dir, 'result.csv'), index=False)
    
    accuracy = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=['normal', 'abnormal'], zero_division=0)
    conf_matrix = confusion_matrix(labels, predictions)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
        f.write("Arguments:\n")
        f.write(f"Theta: {args.theta}\n")
        f.write(f"Smooth Window Size: {args.smooth_window_Hz}\n")
        f.write(f"Threshold Count: {args.threshold_cnt}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
    
    print('Evaluation completed in {:.3f} secs.'.format(time() - start_time))