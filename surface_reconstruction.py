import os
import multiprocessing as mp
from tqdm import tqdm
import argparse

# define a template bash command that will be run by process.
# this command will be run in the shell
command = "splashsurf reconstruct {} -o {} -q -r={} -l={} -c=0.5 -t=0.6 --subdomain-grid=on --mesh-cleanup=on --mesh-smoothing-weights=on --mesh-smoothing-iters=25 --normals=on --normals-smoothing-iters=10"


def process_frame(frame_dir, args):
    files = os.listdir(frame_dir)
    # find file that ends with ply
    ply_list = [f for f in files if f.endswith(".ply")]

    for ply_file in ply_list:
        # note here we do reconstruction for each fluid object seperatedly. 
        # you might need to reconstruct some fluid objects together as a single object. Modify the code here if needed.
        ply_path = os.path.join(frame_dir, ply_file)
        output_path = ply_path.replace(".ply", ".obj")
        # run the command
        os.system(command.format(ply_path, output_path, args.radius, args.smoothing_length))

def worker(frame_dir, args):
    try:
        process_frame(frame_dir, args)
    except Exception as e:
        print(f"failed to process {frame_dir}")
        print(e)
    return 1 # return 1 to indicate success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--radius', type=float, default=0.01)
    parser.add_argument('--smoothing-length', type=float, default=3.5)

    args = parser.parse_args()
    frame_list = os.listdir(args.input_dir)
    frame_list.sort(key=lambda x: int(x))
    num_frames = len(frame_list)


    # Using a pool of workers to process the images
    pool = mp.Pool(args.num_workers)

    # Progress bar setup
    pbar = tqdm(total=len(frame_list))

    # Update progress bar in callback
    def update_pbar(result):
        pbar.update(1)


    for frame in frame_list:
        frame_dir = os.path.join(args.input_dir, frame)
        pool.apply_async(worker, args=(frame_dir, args), callback=update_pbar)

    
    pool.close()
    pool.join()
    pbar.close()



    pool.join()
