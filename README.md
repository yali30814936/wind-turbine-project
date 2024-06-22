# Usage Instructions
This program is designed to evaluate the performance of an anomaly detection algorithm. It also supports converting data files into a format that can be used by this program.

## Important Note
For develepment convenience, I changed the format of the spectrum dataset. The convert feature is also provided. See the example bellow.
**It is mandatory to convert the dataset before evaluating.**

## Dependencies
* pandas
* scikit-learn
* tqdm
* matplotlib

## Program Parameters

* `mode`: The mode of the program, with two options:
  * `evaluate`: Evaluation mode
  * `convert`: Conversion mode


## Optional Parameters
* `-h` or `--help`: Show help message and exit.
* `-i` or `--input_dir`: Input directory, default is `data`.
* `-o` or `--output_dir`: Output directory, default is `results`.
* `-t` or `--theta`: Threshold value, type is float, default is `30`.
* `-s` or `--smooth_window_Hz`: Smooth window size in Hz, type is float, default is `5`.
* `-c` or `--threshold_cnt`: Threshold count, type is integer, default is `3`.
* `-n` or `--ncpu`: Number of CPUs to use, set to `-1` to use half of your cores, type is integer, default is `-1`.
* `-q` or `--quiet`: Quiet mode, use this flag to disable outputting plots for each predictions.
* `f` or `force`: Force to overwrite the output directory if it already exists.


## Usage Examples
### Conversion Mode
You don't have to worry about subdirectories. This program will find them.
```
python main.py convert -i path/to/input -o path/to/output -n 2
```
### Evaluation Mode
You should put the datasets in the following manner:
```
+ data
 + abnormal
 + normal
 + pattern  <- put the sample to represent "normal case" here.
```
Again, **make sure you converted the dataset before evaluating!!!**
```
python main.py evaluate -i path/to/input -o path/to/output -t 35.5 -c 5 -n 4
```
