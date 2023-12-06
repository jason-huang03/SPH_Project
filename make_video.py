import imageio.v2 as imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--fps', type=int, default=20)
args = parser.parse_args()

file_name_list = os.listdir(args.images_dir)
# sort the file
file_name_list.sort(key=lambda x: int(x.split('.')[0]))
images = []
for filename in file_name_list:
    file_path = os.path.join(args.images_dir, filename)
    images.append(imageio.imread(file_path))
imageio.mimsave(args.output_path, images, fps=args.fps)
