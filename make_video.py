import imageio.v2 as imageio
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help="experiement directory")
parser.add_argument('--image_name', type=str, default='raw_view.png')
parser.add_argument('--output_path', type=str, required=True, help="output video path")
parser.add_argument('--fps', type=int, default=20)

args = parser.parse_args()

frame_list = os.listdir(args.input_dir)
# sort the file
frame_list.sort(key=lambda x: int(x))

images = []
for frame in frame_list:
    file_path = os.path.join(args.input_dir, frame, args.image_name)
    try:
        images.append(imageio.imread(file_path))
    except:
        print(f"failed to load image from frame {frame}")

imageio.mimsave(args.output_path, images, fps=args.fps)
