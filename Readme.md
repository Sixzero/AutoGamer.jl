# AutoGamer.jl

A simple package to automate UI usage and games on Android using Julia.

- Collecting gameplay data (recording gameplay video and touches on the screen)
- Preprocess the collected data (extracting frames from the video, and matching them with touches)
- Train a model to play the game
- Run the model to play the game

Still, the package is unpolished, but it works and can be a good base for someone facing similar problems.

## Realtime example

https://github.com/Sixzero/AutoGamer.jl/assets/394602/8b039ecd-7db3-4000-8154-251602a5b16b
Trained on 25 minutes of gameplay. Navigating between parts of the map is not working, sometimes it looks like it would follow some units on the map, but of course, it can be improved easily. 

## Installation

```julia
using Pkg
Pkg.add("https://github.com/Sixzero/AutoGamer.jl.git")
```

## Usage

In the run folder:
- `julia 1.collect.jl` to collect data. You need to modify, the game name(save folder under data/).
- `julia 2.preprocess.jl` to preprocess the data. You need to modify the saved file name you want to work with.
- `julia 3.main.jl` to train the model. The train data will be what the preprocessing script saved.
- `julia 4.run.jl` to run the model. You need a device available for the `adb`, and probably things will work out of the box.

## Train

The train will give a somewhat reliable model, which will predict touches and drags on screen. 

The model is a CNN with a self-attention layer... it is just for testing. It is not the best model for the task, but it is a good start.

## Dependencies

The model uses `Scrcpy.jl` to control the device with low latency (1-5ms). The model inference should be fast enough to work on 15FPS, which I think is bearable for most games.

## TODO

- Better model, or somehow better training loss.
- Collecting data could improve results... although I am not sure. 
- Preprocessing identifies the recorded video start unix time from file birth date, probably there could be better ways to do this.
 - Check whether the preprocessing frames are matched up with the touches not just on the first parts of the video, but in the end they also needs to match.
