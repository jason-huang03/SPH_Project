#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python run_simulation_far.py --scene data/scenes/final_scene1.json
python surface_reconstruction.py --input_dir final_scene1_output --num_workers 3 --smoothing-length 3.2