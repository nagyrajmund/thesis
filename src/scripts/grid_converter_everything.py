import operator
from functools import reduce
from tqdm import tqdm
from os import listdir, makedirs
from os.path import join, isdir, isfile
from shutil import rmtree
from argparse import ArgumentParser
import subprocess

RUN_NAMES = [
    "random baseline, linear-input interpolation, 60 steps, 30 bins",
    "white baseline, linear-input interpolation, 60 steps, 30 bins",
    "black baseline, linear-input interpolation, 60 steps, 30 bins",
    #"black baseline, linear-input interpolation, 30 steps, 30 bins",
    #"black baseline, linear-input interpolation, 30 steps, 50 bins",
]

parser = ArgumentParser()
parser.add_argument("n_max_imgs", type=int, default=10)
parser.add_argument("step", type=int, default = 5)
args = parser.parse_args()

result_dir = join(".", "outputs")
run_names = sorted(listdir(result_dir))

# Create save dir 
save_dir = join(result_dir, "imagemagick")
if isdir(save_dir):
    rmtree(save_dir)
makedirs(save_dir)c


# ----------------------------------------
# Create stacked plots for each run      -
# ----------------------------------------
for run_name in tqdm(RUN_NAMES):
    curr_folder = join(result_dir, run_name)
    curr_heatmap_dir = join(curr_folder, "heatmaps")
    curr_curve_dir = join(curr_folder, "curves")
    curr_save_dir = join(result_dir, "imagemagick", "runs", run_name)

    makedirs(join(curr_save_dir, "combined"))
    makedirs(join(curr_save_dir, "stacked"))
    
    filenames = sorted(listdir(curr_heatmap_dir))[:args.n_max_imgs]

    
    for start_idx in range(0, len(filenames), args.step):
        # -----------------------------------------------------------
        # Combine IG heatmap and insertion/deletion curves per file -
        # -----------------------------------------------------------
        combine_command = lambda idx : [
            "convert",
                f"{join(curr_heatmap_dir, filenames[idx])}",
                f"{join(curr_curve_dir, filenames[idx])}",
                "-resize", "978x491",
                "+append",
                f"{join(curr_save_dir, 'combined', filenames[idx])}"
            ]
        
        for idx in range(start_idx, start_idx + args.step):
            subprocess.run(combine_command(idx))

        # -----------------------------------------------------------
        # Stack combined plots of all 'args.step' files             -
        # -----------------------------------------------------------
        all_combined_images = [
            f"{join(curr_save_dir, 'combined', filenames[idx])}"
            for idx in range(start_idx, start_idx + args.step)
        ]

        stack_command = ["convert"] + \
            ["-pointsize", "40", f"label:'{run_name}'", "-gravity", "center"] + \
            all_combined_images + \
            ["-append"] + \
            [f"{join(curr_save_dir, 'stacked', filenames[start_idx])}"]
            

        subprocess.run(stack_command)

# Combine the plots for different runs, creating a 2D grid of different runs and inputs

filenames = sorted(listdir(join(result_dir, "imagemagick", "runs", RUN_NAMES[0], "stacked")))

for filename in tqdm(filenames):
    stacked_images_from_all_runs = [
        join(result_dir, "imagemagick", "runs", run_name, "stacked", filename)
        for run_name in RUN_NAMES
    ]
    
    final_command = \
        ["convert"] + \
            stacked_images_from_all_runs + \
            ["+append"] + \
            [join(result_dir, "imagemagick", filename)]
    
    subprocess.run(final_command)
    
