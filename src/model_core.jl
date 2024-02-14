
# Define a simple convolutional block
function ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
  Chain(
      Conv((kernel_size, kernel_size), in_channels => out_channels, stride = stride, pad = padding, relu),
      BatchNorm(out_channels, relu)
  )
end

# Self-attention layer
struct SelfAttention
  query::Dense
  key::Dense
  value::Dense
end

function SelfAttention(channels::Int)
  SelfAttention(
      Dense(channels, channels),
      Dense(channels, channels),
      Dense(channels, channels)
  )
end
Flux.trainable(m::SelfAttention) = (query=m.query, key=m.key, value=m.value)
mymul(a, xT) = a.weight * xT .+ a.bias
function (sa::SelfAttention)(x)

    # x is now 2D: (features, batch_size)
    
    
    # Compute query, key, value
    # @edit sa.query(x)
    q = mymul(sa.query, x)
    k = mymul(sa.key, x)
    v = mymul(sa.value, x)
    
    # # Calculate attention scores
    # # scores = softmax(k * q', dims=1)  # Transpose q to match dimensions
    scores = softmax(q .* k, dims=1)
    
    # # Apply attention scores to the values
    # # att_output = v * scores  # Multiply value by scores
    att_output = v .* scores  # Multiply value by scores
    
    # No need to reshape, output is already 2D: (features, batch_size)
    return att_output

end

Flux.@functor SelfAttention

# Example model with attention
function create_attention_model()
  model = Chain(
      ConvBlock(3, 16, 3, 1, 1),  # Example convolutional block
      SelfAttention(16),  # Self-attention layer
      GlobalMeanPool(),  # Pooling layer
      Flux.flatten,
      Dense(16, 10),  # Classification layer for 10 classes
      softmax
  )
  return model
end
