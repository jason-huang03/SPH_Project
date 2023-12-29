#!/bin/bash

python run_simulation_far.py --scene data/scenes/final_scene2.json
python surface_reconstruction.py --input_dir final_scene2_output --num_workers 2