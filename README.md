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

## Usage
```
python main.py <mode> [-h] [-i INPUT_DIR] [-o OUTPUT_DIR] [-t THETA] [-c THRESHOLD_CNT] [-n NCPU] [-s SMOOTH_WINDOW_HZ] [-q] [-f]
```

## Required Parameters
* `mode`: The mode of the program, with two options:
  * `evaluate`: Evaluation mode
  * `convert`: Conversion mode


## Optional Parameters
| Parameter | Description |
| --- | --- |
| `-h` or `--help` | Show help message and exit. |
| `-i` or `--input_dir` | Input directory, default is `data`. |
| `-o` or `--output_dir` | Output directory, default is `results`. |
| `-t` or `--theta` | Threshold value, an RoI grid is considered abnormal if its error is greater or equal to this threshold. Default is `35.0`. |
| `-c` or `--threshold_cnt` | Threshold count, a sample is considered abnormal if the number of abnormal grids is greater or equal to this threshold. Default is `4`. |
| `-n` or `--ncpu` | Number of CPUs to use, set to `-1` to use half of your cores, default is `-1`. |
| `-s` or `--smooth_window_Hz` | Moving average window size in Hz, default is `5.0`. |
| `-q` or `--quiet` | Quiet mode, use this flag to disable outputting plots for each predictions. |
| `-f` or `force` | Force to overwrite the output directory if it already exists. |


## Usage Examples
### Conversion Mode
You don't have to worry about subdirectories. This program will find them.
```
python main.py convert -i path/to/input -o path/to/output -n 2
```
### Evaluation Mode
```
python main.py evaluate -i path/to/input -o path/to/output -t 35.5 -c 5 -n 4
```
You should put the datasets in the following manner:
```
data/
├─ abnormal/
│  └─ abnormal_cases.csv
├─ normal/
│  └─ normal_cases.csv
└─ pattern/
   └─ model_cases.csv    <- put the sample to represent "normal case" here.
```
Again, **make sure you converted the dataset before evaluating!!!**
