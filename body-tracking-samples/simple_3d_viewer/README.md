# Azure Kinect Body Tracking Simple3dViewer Sample

## Installation
### Linux
```
curl -sSL https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod
sudo apt-get update

sudo apt-get install -y ninja-build
sudo apt install k4a-tools
sudo apt install libk4a1.4-dev
```

copy file `https://gist.github.com/madelinegannon/c212dbf24fc42c1f36776342754d81bc`

`scripts/99-k4a.rules` into `/etc/udev/rules.d/`

Edit /etc/default/grub.
Locate the following line:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
```

Replace it by using this line:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash usbcore.usbfs_memory_mb=512"
```
```
sudo update-grub
restart
```

update code and build
```
git pull --recurse-submodules
mkdir build
cd build
cmake .. -GNinja
ninja
```

#### Eigen
download from:
`http://eigen.tuxfamily.org/index.php?title=Main_Page#Download`

copy into place:
```
sudo cp -r eigen-3.4-rc1 /usr/local/eigen3
```

#### tensorrt

install cuda and tensorrt manually. DONT DO DEB!!!
https://developer.nvidia.com/tensorrt-getting-started
cuda 13.1: https://developer.nvidia.com/cuda-11-3-1-download-archive

## Introduction

The Azure Kinect Body Tracking Simple3dViewer sample creates a 3d window that visualizes all the information provided
by the body tracking SDK.

## Usage Info

USAGE: simple_3d_viewer.exe SensorMode[NFOV_UNBINNED, WFOV_BINNED](optional) RuntimeMode[CPU, OFFLINE](optional)
* SensorMode:
  * NFOV_UNBINNED (default) - Narraw Field of View Unbinned Mode [Resolution: 640x576; FOI: 75 degree x 65 degree]
  * WFOV_BINNED             - Wide Field of View Binned Mode [Resolution: 512x512; FOI: 120 degree x 120 degree]
* RuntimeMode:
  * CPU - Use the CPU only mode. It runs on machines without a GPU but it will be much slower
  * OFFLINE - Play a specified file. Does not require Kinect device. Can use with CPU mode

```
e.g.   simple_3d_viewer.exe WFOV_BINNED CPU
                 simple_3d_viewer.exe CPU
                 simple_3d_viewer.exe WFOV_BINNED
                 simple_3d_viewer.exe OFFLINE MyFile.mkv
```

## Instruction

### Basic Navigation:
* Rotate: Rotate the camera by moving the mouse while holding mouse left button
* Pan: Translate the scene by holding Ctrl key and drag the scene with mouse left button
* Zoom in/out: Move closer/farther away from the scene center by scrolling the mouse scroll wheel
* Select Center: Center the scene based on a detected joint by right clicking the joint with mouse

### Key Shortcuts
* ESC: quit
* h: help
* b: body visualization mode
* k: 3d window layout


