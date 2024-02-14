module Data

using Boilerplate: @typeof
using Flux
using Images
using Glob
using Statistics
using CSV
using DataFrames
using Clustering
using Base.Threads
using Boilerplate
using ProgressBars

export get_train_data, get_x2, create_inputs, split_data, get_screen_norm, get_max_norm, convert_coords_to_clusters, plot_convert_coords, denorm, soft_denorm, MeanNorm, MeanIntNorm, ClusterNorm, MixBatchNorm, apply_norm, merge_norms, MixBatchNorm
  # @time begin
  # v = channelview(img)
  # r = rawview(v) |> Array
  # r = r ./ 255
  # r = mean(r, dims=1)[1,:,:]
  # end
read_file(f::String) = begin
  img = load(f)::Matrix{RGB{N0f8}}
  img_small = imresize(img, (108,108)) # 28x28 for mnist
  gray_img = Gray.(img_small)
  normalized_img = [gray.val for gray in float(gray_img)]
  return normalized_img
end

function get_train_data(game, ratio=0.1)
  data = CSV.read("data/$game/data_map.csv", DataFrame)
  # initialize imgs as a vecotr of Matrix{Float32}  
  imgs = Vector{Matrix{Float32}}(undef, length(data.file))
  # read files and show the progress
  # progress = (length(data.file))
  @time Threads.@threads for i in ProgressBar(1:length(data.file), printing_delay=0.01)
    imgs[i] = read_file(data.file[i])
    # next!(progress)
  end
  x = stack(imgs, dims=3)
  y = stack([data.x,data.y, data.start_x - data.x, data.start_y - data.y], dims=1)
  x = Flux.unsqueeze(x, 3) # give back the RGB channel as one grayscale channel

  return Float32.(x), y
end
function get_screen_norm(x, _::Val{:POCO})
  norm = reshape([1080, 2400], 2, 1)
  @assert all(norm*0.7 .<= maximum(x, dims=2)) "We probably scale with wrong factor: Maxes: $(maximum(x, dims=2)) Screensize: $(norm)"
  return x ./ norm, MeanNorm(norm)
end
function get_max_norm(x)
  norm = maximum(x, dims=2)
  @show norm
  x_norm = x ./ norm
  return x_norm, MeanNorm(norm)
end
function get_x2(y, input_depth, permute_idxs, selected_inputs=3:4)
  len = length(selected_inputs)
  x2 = zeros(Float32, len * input_depth, length(permute_idxs))
  for i in eachindex(permute_idxs)
    idx = permute_idxs[i]
    max_input_depth = min(idx-1-1,input_depth)
    x2[end-len*max_input_depth+1:end,i] .= reshape(y[selected_inputs,idx-1-max_input_depth+1:idx-1],:)
  end
  x2
end

function create_inputs(x, y, input_depth)
  x2 = get_x2(y, input_depth, 1:size(y,2))
  x,x2,y
end
function split_data(x, x2, y, ratio=0.1, shuffle=true)
  B = size(y, 2)
  permute = (1:B) # Keep in mind
  if shuffle
    permute = shuffle(permute)
  end 
  splitpoint = Int(floor(B*ratio))
  permute_test, permute_train = permute[1:splitpoint], permute[splitpoint+1:end]
  x_train, x_test = x[:,:,:,permute_train], x[:,:,:,permute_test]
  x2_train, x2_test = x2[:,permute_train], x2[:,permute_test]
  y_train, y_test = y[:,permute_train], y[:,permute_test]
  return (x_train, x_test), (y_train, y_test), (x2_train, x2_test), (permute_train, permute_test)
end
using Clustering
function convert_coords_to_clusters(coords, k=9)
  result = kmeans(coords, k)  # 'result' is an object containing clustering results
  # Cluster centers
  centers = result.centers
  # Assignments of points to clusters
  assignments = result.assignments

  y = Array(Flux.onehotbatch(assignments, 1:k))
  Float32.(y), ClusterNorm(centers, k, result.totalcost)
end
function plot_convert_coords(y, range=2:40)
  plot(k -> kmeans(y, k).totalcost+1f-7, range, yscale=:log10)
end
struct ClusterNorm
  centers
  k
  totalcost
end
struct MeanNorm
  mean
end
struct MeanIntNorm
  mean
end
# denormalize coord clusters
function denorm(norm::ClusterNorm, y)
  # onehot to onecold
  y_idx = Flux.onecold(y)
  # onecold to centers
  norm.centers[:,y_idx]
end
using Distributions
# denormalize coord clusters
function soft_denorm(norm::ClusterNorm, probs::Matrix)
  idxs = [rand(Categorical(probs[:,i])) for i in 1:size(probs,2)]
  norm.centers[:,idxs]
end
function soft_denorm(norm::ClusterNorm, probs::Vector)
  dist = Categorical(probs)
  y_idx = rand(y)
  # onecold to centers
  norm.centers[:,y_idx]
end
Base.size(norm::ClusterNorm) = size(norm.centers, 2)
function denorm(norm::MeanNorm, y)
  y = y .* norm.mean
end
denorm(norm::MeanIntNorm, y, y2) = (Int.(round.(y .* norm.mean)), Int.(round.(y2 .* norm.mean)))
denorm(norm::MeanNorm, y, y2) = (y .* norm.mean, y2 .* norm.mean)
apply_norm(normer::MeanNorm, x) = x ./ normer.mean
# MeanNorm(img::Matrix) = MeanNorm(size(img)[1:2])

function merge_norms(norm1::MeanNorm, norm2::MeanNorm)
  MeanNorm(norm1.mean * norm2.mean)
end
struct MixBatchNorm
  dict::Dict
end
# A denormer on the first batch dimension
function denorm(normer::MixBatchNorm, x)
  idx = 1
  res = []
  for (key, norm) in (normer.dict)
    len = size(norm)
    push!(res, denorm(norm, x[idx:idx+len-1,:]))
    idx += len
  end
  # vcat(Tuple(denorm(norm, x[:,:]) for (key, norm) in (normer.dict))...)
  vcat(res...)
end

function soft_denorm(normer::MixBatchNorm, x)
  idx = 1
  res = []
  for (key, norm) in (normer.dict)
    len = size(norm)
    push!(res, soft_denorm(norm, x[idx:idx+len-1,:]))
    idx += len
  end
  # vcat(Tuple(denorm(norm, x[:,:]) for (key, norm) in (normer.dict))...)
  vcat(res...)
end

end # module Data