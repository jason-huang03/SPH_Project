# SPH Realization of Fluid Simulation

## Features

### ✨ SPH Algorithm

+ [Weakly compressible SPH for free surface flows (WCSPH)](https://dl.acm.org/doi/10.5555/1272690.1272719)
+ [Predictive-Corrective Incompressible SPH (PCISPH)](https://dl.acm.org/doi/10.1145/1576246.1531346)
+ [Position Based Fluids (PBF)](https://dl.acm.org/doi/10.1145/2461912.2461984)
+ [Divergence-Free Smoothed Particle Hydrodynamics (DFSPH)](https://dl.acm.org/doi/10.1145/2786784.2786796)

### ✨ Fluid-Rigid Interaction Force

+ [Versatile Rigid-Fluid Coupling for Incompressible SPH](https://dl.acm.org/doi/10.1145/2185520.2185558)

### ✨ Viscosity

+ [Standard Viscosity](https://iopscience.iop.org/article/10.1088/0034-4885/68/8/R01)
+ [A Physically Consistent Implicit Viscosity Solver for SPH Fluids](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.13349)

### ✨ Rigid Solver

+ [Bullet Physics Engine](https://github.com/bulletphysics/bullet3)

### ✨ SPH Surface Reconstruction

+ [Splash Surf](https://github.com/InteractiveComputerGraphics/splashsurf)

### ✨ Rendering

+ [Blender](https://www.blender.org/)

## Installation on Linux

### Python Environment

```bash
git clone https://github.com/jason-huang03/SPH_Project.git
cd SPH_Project
conda create --name SPH python=3.9
conda activate SPH
pip install -r requirements.txt
```

The code is tested on Ubuntu 22.04, Python 3.9.12, CUDA 12.2 with NVIDIA A100 GPU.



### Install Vulkan SDK (Optional)

You may need Vulkan SDK to run the code. Here we provide a way to install Vulkan SDK on Linux without admin permission. 

```bash
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz -O vulkan-sdk.tar.gz


sudo mkdir -p /opt/vulkan-sdk
sudo tar -xvf vulkan-sdk.tar.gz -C /opt/vulkan-sdk # You can extract to your customized place. Change following lines accordingly.

VULKAN_SDK_VERSION=$(ls /opt/vulkan-sdk/ | grep -v "tar.gz")

echo "VULKAN_SDK_VERSION: $VULKAN_SDK_VERSION" # should be something like 1.3.268.0
```

Then you can add the following lines in your `~/.bashrc` file.

```bash
# add following line in your ~/.bashrc
# suppose VULKAN_SDK_VERSION has a value

export VULKAN_SDK=/opt/vulkan-sdk/$VULKAN_SDK_VERSION/x86_64
export PATH="$PATH:$VULKAN_SDK/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VULKAN_SDK/lib"

# possibly you will need this line
export VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/explicit_layer.d"

```

After that, you can check if you have successfully installed Vulkan SDK by running `source ~/.bashrc` followed by `vulkaninfo` in your terminal.



### Install Splash Surf (Optional)

You can refer to the [official document](https://github.com/InteractiveComputerGraphics/splashsurf) of splashsurf for more detail. Here we provide a way to install splashsurf on Linux without admin permission.

```bash
# install rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# install splashsurf
cargo install splashsurf
```



### Install Blender (Optional)

You can refer to the [official website](https://www.blender.org/) of blender for more detail. Here we provide a way to install blender on LInux without admin permission.

```bash
# download blender 3.6 Linux package from https://www.blender.org/download/lts/3-6/

# uncompressed the .tar.gz file
tar -xf blender-3.6.7-linux-x64.tar.xz
```

Add the following line in your `~/.bashrc` file.

```bash
# update the $PATH variable
# add the following line in ~/.bashrc file
export PATH=$PATH:~/blender-3.6.7-linux-x64/
```

The rendering script is tested with blender 3.6.7 and blender 4.0 seems uncompatible.



## Usage

For a quick start, try the following command and make sure you turn on the export settings in the `json` scene configuration.

```bash
python run_simulation.py --scene ./data/scenes/dragon_bath_dfsph.json
```

To visualize the results, you can run the following command to make the images into a video. Those raw images is derived from Taichi GGUI API.

```bash
python make_video.py --input_dir ./dragon_bath_dfsph_output \
--image_name raw_view.png --output_path --video.mp4 --fps 30
```

To make the `.ply` particle file into `.obj` file for rendering, you can do surface reconstruction with the following command:

```bash
python surface_reconstruction.py --input_dir ./dragon_bath_dfsph_output --num_workers 2
```

This will open `num_workers` processes to do surface reconstruction with [splashsurf](https://github.com/InteractiveComputerGraphics/splashsurf).

Then you can do rendering with [blender](https://www.blender.org/). We suggest you to first make a scene with a graphical user interface, setting materials, lighting, camera, rendering parameters and add other static objects. Then you can save the scene as a `.blend` file. With this, you can render the whole simulation process by running

```bash
CUDA_VISIBLE_DEVICES=0 python render.py --scene_file ./scene.blend \
--input_dir ./dragon_bath_dfsph_output --num_workers=1 --device_type OPTIX
```

The rendering script can put rendering jobs on multiple gpus.  You can set `CUDA_VISIBLE_DEVICES` and `num_workers` according to your available devices.



## Future Work

+ Correct implementation of [Implicit Incompressible SPH](https://ieeexplore.ieee.org/document/6570475)
+ Integration of [Position Based Fluids](https://dl.acm.org/doi/10.1145/2461912.2461984)
+ Strong rigid-fluid coupling following [Interlinked SPH Pressure Solvers for Strong Fluid-Rigid Coupling](https://dl.acm.org/doi/10.1145/3284980)




## Acknowledgements

This project is built upon the following repositories:

+ [Taichi](https://github.com/taichi-dev/taichi)
+ [Bullet](https://github.com/bulletphysics/bullet3)
+ [SPH Taichi](https://github.com/erizmr/SPH_Taichi)
+ [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
+ [Splash Surf](https://github.com/InteractiveComputerGraphics/splashsurf)
+ [Blender](https://www.blender.org/)

We thank all contributors of these repositories for their great work and open source spirit.