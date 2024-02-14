includet("draw.jl")

using Images
using Boilerplate
using Scrcpy
# using ImageView
# using GLMakie

function get_val(arr) 
  [col.val for col in float(arr)]
end
function convert2rgb(arr::Array{UInt8,3})
  height, width, _ = size(arr)
  return [RGB{N0f8}(arr[y, x, 1] / 255, arr[y, x, 2] / 255, arr[y, x, 3] / 255) for y in 1:height, x in 1:width]
end
function main_loop(video_socket, state, resolution=(1200, 544), x2_dims=3:4)
    # Open the video capture device
    cap = get_stream()

    m = fmap(cu, model)

    # win = ImageView.imshow(rand(RGB, 640, 480))

    # Create a figure for display
    # fig = Figure(resolution = (432, 432))
    # ax = Axis(fig[1, 1]; aspect = :equal)
    # hidedecorations!(ax); hidespines!(ax)
    dbg_img_size = resolution
    # img = Observable(rand(RGB, dbg_img_size...))
    # @sizes img[]

    # imgplot = image(@lift(rotr90($img)), axis = (aspect=DataAspect(),), figure = (figure_padding=0, resolution=(dbg_img_size[2], dbg_img_size[1])))
    # hidedecorations!(imgplot.axis)
    # display(imgplot)

    # Display the first frame
    # imshow!(ax, image)
    dt_avg = 0f0
    input_depth = 60
    X2_size = length(x2_dims)
    x2 = zeros(Float32, X2_size * input_depth, 1)
    try
    while true

      dt = @elapsed begin
        # Read a frame from the device
        frame = read(cap)
        # @time frames = get_next_frames(video_socket, 0xa222)
        # if frames === nothing
        #   continue
        # end
        # frame = convert2rgb(frames[end])
        frame_small = imresize(frame, (108,108))
        # converting frame_small to an array of floats
        frame_gray = Gray.(frame_small)
        frame_array = get_val(frame_gray)

        # ImageView.imshow(win, frame_array)
        # Update the observable image
        # Display the frame for testing (optional)
        # imshow("Frame", frame_array)
        # create_inputs(x, y, input_depth)
        frame_array = reshape(frame_array, 108, 108, 1, 1)
        frame_dbg = imresize(frame, dbg_img_size...)
        frame_array_cu, x2_cu = cu(frame_array), cu(x2)
        res = cpu(m(frame_array_cu, x2_cu))
        # @show sort(res[51:100])
        coords_f = soft_denorm(normer_mix, res)
        coords_f = vcat(coords_f[2:-1:1,1:1], coords_f[4:-1:3,1:1])
        @show coords_f
        c, Δc = denorm(state, coords_f[1:2,1], coords_f[3:4,1])
        c_img, Δc_img = coords_f[1:2,1] .* dbg_img_size, coords_f[3:4,1] .* dbg_img_size
        @show c, Δc, state.resolution
        @show c_img, Δc_img, dbg_img_size
        set_swipe(state, c[1] + Δc[1], c[2] + Δc[2], Δc[1], Δc[2])
        draw_dot!(frame_dbg, PointRound((c_img .+ Δc_img)...), RGB{Float32}(0.5,0,0), 10)
        draw_dot!(frame_dbg, PointRound(c_img...), RGB{N0f8}(1,0,0), 20)
        # img[] = frame_dbg
        # shift x2
        x2[1:end-X2_size,:] .= x2[X2_size+1:end,:] 
        x2[end-X2_size+1:end,:] .= coords_f[x2_dims,:]
      end
      # calc average fps:
      dt_avg = 0.8 * dt_avg + 0.2 * dt
      println("FPS: ", 1/dt_avg)
    end
    # only catch InterruptException
    catch e
      close(cap)
      if e isa InterruptException
        # cleanup
        println("function terminated by user")
      else
        rethrow(e)
        # do something else
      end
    end
end

device::String = "/dev/video0"
video_socket, ctrl_socket, state = initialize_scrcpy_control(true, 1200)
main_loop(device, state)
# capture_frames(video_socket)
#%%
# video_socket, ctrl_socket, state = initialize_scrcpy_control(true, 1200)

end_touch(state)
#%%
set_swipe_position(state, 250, 250, -100, 100)
#%%
for i in 1:100000
@time frames = get_next_frames(video_socket, 0x2222)
end
#%%
using Images
using ImageView

for i in 1:5
  frames = get_next_frames(video_socket, 0x2222)
  if frames !== nothing
    @time frame = convert_to_rgb(frames[end])
    @display frame
  end
end
#%%
using Images

# Example to create a sample image
width, height = 100, 100  # Define the dimensions of the image
image = rand(RGB{N0f8}, height, width)  # Create a random RGB image
@typeof image

# Display the image inline
display(image)
#%%
using Distributions

function get_idx(values, probs)
    # Normalize logits to probabilities
    # probs = exp.(logits) / sum(exp.(logits))
    
    # Create a categorical distribution with the given probabilities
    dist = Categorical(probs)
    
    # Sample an index from the distribution
    idx = rand(dist)
    
    return values[idx]
end

values = [1,2,3]
logits = [0.2,0.7,0.1]
# get_idx based no logits probabilities
idx = get_idx(values, logits)
