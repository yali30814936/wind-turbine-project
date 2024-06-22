import argparse
from src.evaluate import evaluate
from src.convert import convert

def parameter():
    parser = argparse.ArgumentParser(description='Evaluate the performance of the abonomaly detection algorithm. It also supports converting the data files to a format that can be used by this program.')
    parser.add_argument('mode', choices=['evaluate', 'convert'], help='Mode of the program')
    parser.add_argument('-i', '--input_dir', help='Input directory', default='data')
    parser.add_argument('-o', '--output_dir', help='Output directory', default='results')
    parser.add_argument('-t', '--theta', help='Threshold value', type=float, default=30)
    parser.add_argument('-s', '--smooth_window_Hz', help='Smooth window size in Hz', type=float, default=5)
    parser.add_argument('-c', '--threshold_cnt', help='Threshold count', type=int, default=3)
    parser.add_argument('-n', '--ncpu', help='Number of CPUs to use. Set to -1 to use half of your cores', type=int, default=-1)
    parser.add_argument('-q', '--quite', help='Disable drawing plots for each predictions', action='store_true')
    parser.add_argument('-f', '--force', help='Force to overwrite the output directory', action='store_true')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parameter()
    
    if args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'convert':
        convert(args)
    else:
        raise ValueError('Invalid mode. Choose either evaluate or convert.')