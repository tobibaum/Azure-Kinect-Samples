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

## Shortcuts
* q: quit
* r: start recording
* s: save recording
* l: load recording
* x: unload recording
* d: return to default mode
