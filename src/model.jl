module Model

using Boilerplate: @typeof
using Flux
using Statistics

using Random
using Boilerplate
using LinearAlgebra
include("model_core.jl")

export create_model
# @sizes Flux.onehotbatch(vcat(zeros(length(yF)), ones(length(yT))), 0:1)
lrelu(x) = max.(0.05*x, x)

function scaled_glorot_uniform(shape...)
  scale = 0.5
  return scale * Flux.glorot_uniform(shape...)
end
function get_weight((in, out)::Pair{<:Integer, <:Integer})
	# res = reshape(collect(((1:in*out) ./ Float32((in*out)) .- 0.5f0)/1f0), out, in)
		res = Flux.glorot_uniform(out, in) .* 0.01
	# @show res
	res
end 

struct ModelCombine
  frame_branch::Chain
  touch_branch::Chain
  output_model
end

Flux.trainable(m::ModelCombine) = (frame_branch=m.frame_branch, touch_branch=m.touch_branch, output_model=m.output_model)
Flux.@functor ModelCombine # this was necessary to create this: _trainable(::Tuple{}, ::@NamedTuple{Wxx::Matrix{Float64}, Wend::Matrix{Float64}}) for the setup(opt, model) fn.
function (m::ModelCombine)(frame_input, touch_input)

  frame_features = m.frame_branch(frame_input)
  touch_features = m.touch_branch(touch_input)

  # Combine features
  combined_features = vcat(frame_features, touch_features)

  # Final processing (e.g., classification layer)
  return m.output_model(combined_features)
end

function create_model(touch_input_size; selected_outputs=3:4, out_size=2, out_size2=2)
  selected_out_size = length(selected_outputs)
  coord_size = 4
  l1_size = 32
  touch_out_size = 40
  layer_touch_size = 10
  layer_coord_size = 10
  frame_branch = Chain(
    make_convs((5, 8, 2, 2, lrelu), # 28x28 => 14x14a
    # (3, 16, 1, 2, lrelu), # 14x14 => 7x7
    # (3, 32, 1, 2, lrelu), # 7x7 => 4x4
    (3, l1_size, 1, 2, lrelu), # 4x4 => 2x2
    )...,
    # Average pooling on each width x height feature map
    GlobalMeanPool(), 
    Flux.flatten,
    SelfAttention(l1_size),  # Self-attention layer
  )
  touch_branch = Chain(
    Dense(touch_input_size*selected_out_size, l1_size), lrelu,
    Dense(l1_size, touch_out_size), lrelu,
    SelfAttention(touch_out_size)
  )

  coord_model = Chain(Dense(l1_size + touch_out_size, layer_coord_size, relu), Dense(layer_coord_size, out_size), softmax)
  drag_model = Chain(Dense(l1_size + touch_out_size, layer_touch_size, relu), Dense(layer_touch_size, out_size2), softmax)
  output_model = Parallel(vcat, coord_model, drag_model)
  m = ModelCombine(frame_branch, touch_branch, output_model)

  # @show Flux.trainable(output_model)
  opt = ADAM(0.0005, (0.8, 0.999))
  # opt = AdaGrad(0.03, )

  # opt = Descent(0.1)
  m, opt
end
make_conv(size, dim, pad, stride, act) = Conv((size, size), dim, init=scaled_glorot_uniform, pad=pad, stride=stride, act)
make_convs(args...) = begin
  [make_conv(size, i == 1 ? (1 => dim) : (args[i-1][2] => dim), pad, stride, act) for (i,(size,dim, pad, stride, act)) in enumerate(args)]
end

end