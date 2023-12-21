import imageio.v2 as imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--fps', type=int, default=20)
args = parser.parse_args()

frame_list = os.listdir(args.images_dir)
# sort the file
frame_list.sort(key=lambda x: int(x))
images = []
for frame in frame_list:
    file_path = os.path.join(args.images_dir, frame, "raw_view.png")
    images.append(imageio.imread(file_path))
    
imageio.mimsave(args.output_path, images, fps=args.fps)
