








# Setting Up scrcpy with v4l2loopback

## Install v4l2loopback:

This step depends on your Linux distribution. On Debian-based systems, you can install it using:

```bash
sudo apt-get install v4l2loopback-dkms
sudo apt-get install v4l-utils
```
## Load the module:
```bash
sudo modprobe v4l2loopback
```
## Identify the Video Device:

After loading v4l2loopback, a new virtual video device will be created (e.g., /dev/video0, /dev/video1, etc.). You can check the device name with:
```bash
v4l2-ctl --list-devices
```
## Use scrcpy with v4l2-sink:

Run scrcpy and specify the virtual video device:
```bash
scrcpy --no-display --v4l2-sink=/dev/video0
```


```
add VideoIO Images ImageView ImageDraw Clustering
```