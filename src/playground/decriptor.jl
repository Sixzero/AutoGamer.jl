using ImageFeatures, TestImages, Images, ImageDraw, CoordinateTransformations, Rotations

img = testimage("lighthouse")
img1 = Gray.(img)
rot = recenter(RotMatrix(5pi/6), [size(img1)...] .÷ 2)  # a rotation around the center
tform = rot ∘ Translation(-50, -40)
img2 = warp(img1, tform, axes(img1))

#%%
features_1 = Features(fastcorners(img1, 12, 0.35))
features_2 = Features(fastcorners(img2, 12, 0.35))
#%%

brisk_params = BRISK()

#%%
desc_1, ret_features_1 = create_descriptor(img1, features_1, brisk_params)
desc_2, ret_features_2 = create_descriptor(img2, features_2, brisk_params)
@show ret_features_2[10]
nothing # hide
#%%
function convert_gray_to_rgba(gray_image::Matrix{Gray{N0f8}})
  return convert(Matrix{RGBA{N0f8}}, gray_image)
end

using Boilerplate
using Colors
img1_copy = convert_gray_to_rgba(img1)
@typeof img1_copy
# @display img1_copy
offset = CartesianIndex(0, 10)

map(ret_features_1) do m 
  mk = m.keypoint
  new_line = LineSegment(mk, mk + offset)
  color = RGBA{N0f8}(1.,0.,1.,0.2)  # color with 20% opacity. 
  draw!(img1_copy, new_line, color)
end
img1_copy
#%%
#%%
# ai"How to index CartesianIndex{2} in julia, I want to get the first index then the second"
res = ai"Convert Matrix{Gray{N0f8}} to Matrix{RGB{N0f8}} in julia, only send me the code snippet"gpt4t
#%%
using PromptingTools: UserMessage
msg = res
new_conversation = vcat(msg, UserMessage("ERROR: ArgumentError: `reinterpret(reshape, RGB{N0f8}, a)` where `eltype(a)` is Gray{N0f8} requires that `axes(a, 1)` (got Base.OneTo(512)) be equal to 1:3 (from the ratio of element sizes)"))
msg2 = aigenerate(new_conversation)
#%%
matches = match_keypoints(Keypoints(ret_features_1), Keypoints(ret_features_2), desc_1, desc_2, 0.1)
nothing # hide
#%%

grid = hcat(img1, img2)
offset = CartesianIndex(0, size(img1, 2))
draw_fn(m) = begin
  @show typeof(m[1])
  @show typeof(grid)
  draw!(grid, LineSegment(m[1], m[2] + offset), )
end
map(draw_fn, matches)
grid
#%%

ai"How to list snap packages"
#%%
ai"How to update a snap package chatgpt-desktop"
#%%
using Pipe
using DataFrames
tmps = @pipe aitemplates("Julia") |> DataFrame |> _[:,5:end]
tmps = @pipe aitemplates("Julia") |> DataFrame |> _[:,[:system_preview]] |> Array |> join(_, "\n") |> show
#%%
aitemplates("JuliaExpertAsk")
#%%

at = AITemplate(:JuliaExpertAsk) 
@show at
dump(at)
# aigenerate(at; ask="Please repeate once what I just said to you, not just this but the pharagrapsh before this:")