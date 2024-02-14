function calculate_output_size(input_size, kernel_size, stride, padding)
  return ((input_size + 2 * padding - kernel_size) / stride) + 1
end

# Test a range of sizes
for input_size in 28:241
# for input_size in 108:108
  size = input_size
  valid = true

  # Apply the formula for each layer
  for i in 1:4  # Assuming 4 convolutional layers
      kernel_size = i == 1 ? 5 : 3
      padding = i == 1 ? 2 : 1
      stride = 2
      size = calculate_output_size(size, kernel_size, stride, padding)

      # Check if size is valid (not too small and not fractional)
      if size < 2 || size != floor(size)
          valid = false
          break
      end
  end

  if valid
      println("Valid input size: ", input_size)
  end
end
#%%
include("draw.jl")
using Images
# using ImageView

count::Int = 0
function draw_img_i(data,i)
  x_train, x2_train, y_train = data
  global count
  # img = Gray.(x_train[:,:,1,1])      # Convert to a grayscale image
  img_colored = RGB.(x_train[:,:,1,i])      # Convert to a grayscale image
  img_colored = imresize(img_colored, (size(img_colored).*3)...)
  x_train_B, y_train_B, x2_train_B = get_data_selection(x_train, y_train, x2_train, i:i)
  nn = MeanNorm(size(img_colored))
  coords_y_f = denorm(normer_mix, y_train_B)
  # @show normer_mix
  y_dot, y_drag = denorm(nn, coords_y_f[1:2,1], coords_y_f[3:4,1])
  green, red = RGB{Float32}(0,1,0), RGB{Float32}(1,0,0)
  p1 = model(x_train_B, x2_train_B)
  coords_f = denorm(normer_mix, p1)
  y_dot, y_drag = y_dot[2:-1:1], y_drag[2:-1:1]
  c, Δc = denorm(nn, coords_f[1:2,1], coords_f[3:4,1])
  c, Δc = c[2:-1:1], Δc[2:-1:1]
  if y_train_B[1,1]!= 1
    count += 1
    # @show count, i, count/i
    # @show Flux.onecold(y_train_B)
  end
  if c[1]>0 || y_dot[1]>0
    @show i
    @show c,  Δc
    @show y_dot, y_drag

    # Display the result
    draw_dot!(img_colored, PointRound((y_dot .+ y_drag)...), RGB{Float32}(0,0.5,0), 15)
    draw_dot!(img_colored, PointRound(y_dot...), green, 15)
    draw_dot!(img_colored, PointRound((c.+Δc)...), RGB{Float32}(0.5,0,0), 10)
    draw_dot!(img_colored, PointRound(c...), red, 10)
    @display img_colored
  end
end
n = size(x_train, 4)
@show n
idxs = 100:2387
idxs = 2387:4000
idxs = 20100:28000
for i in idxs
  draw_img_i((x_train, x2_train, y_train), i)
  # draw_img_i((x_test, x2_test, y_test), i)
end
;
