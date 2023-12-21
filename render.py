import os
import subprocess
import re
import multiprocessing as mp
from tqdm import tqdm
import argparse

def get_visible_gpu_indices():
    # Read the CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if cuda_visible_devices is None:
        # If the environment variable is not set, all GPUs are visible
        return None
    elif cuda_visible_devices.strip() == "":
        # If the environment variable is set to an empty string, no GPUs are visible
        return []
    else:
        # Split the environment variable by comma and convert to integers
        return [int(gpu.strip()) for gpu in cuda_visible_devices.split(',')]

def get_gpu_count():
    try:
        # Get visible GPU indices from the environment variable
        visible_gpu_indices = get_visible_gpu_indices()

        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Decode the output
        output = result.stdout.decode('utf-8')

        # Count the number of lines in the output
        total_gpus = len(re.findall(r'.+\n', output))

        if visible_gpu_indices is None:
            # If no environment variable is set, all GPUs are visible
            return total_gpus
        else:
            # Filter the GPU indices based on the environment variable
            return len([i for i in visible_gpu_indices if i < total_gpus])
    except Exception as e:
        print("An error occurred: ", e)
        return 0



# define a template bash command that will be run by process.
# this command will be run in the shell
command = "blender -b {} --python rendering_script.py -- {} {} {} {} {}" # disable stdout
num_gpus = get_gpu_count()
print("Number of Visible GPUs:", num_gpus)

def process_frame(frame_dir, rank, args):
    os.system(
        command.format(args.scene_file, args.device_type, rank%num_gpus, frame_dir, os.path.join(frame_dir, args.rendered_image_name), "" if (rank == 0 and not args.quiet) else " > /dev/null 2>&1")
    ) # disable stdout on all but the first process

def worker(frame_dir, rank, args):
    try:
        process_frame(frame_dir, rank, args)
    except Exception as e:
        print(f"failed to process {frame_dir}")
        print(e)
    return 1 # return 1 to indicate success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_file', type=str, required=True)
    parser.add_argument('--rendered_image_name', type=str, default='render.png')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device_type', type=str, default='OPTIX')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    frame_list = os.listdir(args.input_dir)
    frame_list.sort(key=lambda x: int(x))
    num_frames = len(frame_list)

    print(f"Processing {num_frames} frames with {args.num_workers} workers")
    print(f"Using device type: {args.device_type}")

    


    # Using a pool of workers to process the images
    pool = mp.Pool(args.num_workers)

    # Progress bar setup
    pbar = tqdm(total=len(frame_list))

    # Update progress bar in callback
    def update_pbar(result):
        pbar.update(1)


    for i, frame in enumerate(frame_list):
        frame_dir = os.path.join(args.input_dir, frame)
        rank = i % args.num_workers
        pool.apply_async(worker, args=(frame_dir, rank, args), callback=update_pbar)

    
    pool.close()
    pool.join()
    pbar.close()



    pool.join()
