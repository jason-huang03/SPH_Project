# SPH Realization of Fluid Simulation

## Paper Implementations

### ✨ SPH Algorithm
+ Weakly compressible SPH for free surface flows (WCSPH)
+ Predictive-Corrective Incompressible SPH (PCISPH)
+ Position Based Fluids (PBF)
+ Divergence-Free Smoothed Particle Hydrodynamics (DFSPH)

### ✨ Fluid-Rigid Interaction Force
+ Versatile Rigid-Fluid Coupling for Incompressible SPH

## Installation

### Python Environment
```bash
git clone https://github.com/jason-huang03/SPH_Project.git
cd SPH_Project
conda create --name SPH python=3.9
conda activate SPH
pip install -r requirements.txt
```

The code is tested on Ubuntu 22.04 with Python 3.9.12, CUDA 12.2 with NVIDIA A100 GPU.

### Install Vulkan SDK
You may need Vulkan SDK to run the code. Here we provide a way to install Vulkan SDK on Linux without admin permission. 

```bash
wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz -O vulkan-sdk.tar.gz
# seems that you can install in you customized place without admin.

sudo mkdir -p /opt/vulkan-sdk
sudo tar -xvf vulkan-sdk.tar.gz -C /opt/vulkan-sdk

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

# possibly you need this line
export VK_LAYER_PATH="$VULKAN_SDK/etc/vulkan/explicit_layer.d"

```

After that, you can check if you have successfully installed Vulkan SDK by running `source ~/.bashrc` followed by `vulkaninfo` in your terminal.

## Usage
For a quick start, try the following command:
```bash
python run_simulation_dfsph.py --scene ./data/scenes/high_fluid_dfsph.json
python run_simulation_dfspc.py --scene ./data/scenes/dragon_bath_dfspc.json
```