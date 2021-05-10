from tqdm import tqdm
from os import listdir, makedirs
from os.path import join
from shutil import rmtree
from argparse import ArgumentParser
import subprocess

parser = ArgumentParser()
parser.add_argument("dir", type=str)
parser.add_argument("-n", type=int, default=3)

args = parser.parse_args()

heatmap_dir = join(args.dir, "heatmaps")
curve_dir = join(args.dir, "curves")
output_dir = join(args.dir, "imagemagick")
rmtree(output_dir)
makedirs(output_dir)
filenames = sorted(listdir(heatmap_dir))

for filename in tqdm(filenames[:2]):
    subprocess.run([
        "convert", 
        join(heatmap_dir, filename), "-resize", "2400x2400",
        join(curve_dir, filename),
        "+append", 
        join(output_dir, filename)
    ])
