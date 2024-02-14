#%%
using AutoGamer
# includet("data.jl")
# includet("model.jl")
# includet("train.jl")

gamename = "alien"

x,y = get_train_data(gamename);
#%%
# y2split, normer = get_max_norm(y)
y_norm, normer = get_screen_norm(y[1:2,:], Val(:POCO))
@show normer.mean
y_normed = vcat(y_norm, apply_norm(normer, y[3:4,:]))
y_normed_p1, normer_p1 = convert_coords_to_clusters(y_normed[1:2,:], 50)
y_normed_p2, normer_p2 = convert_coords_to_clusters(y_normed[3:4,:], 50)
# merge_norms(normer, normer2)
y_normed_cl = vcat(y_normed_p1, y_normed_p2)
normer_mix = MixBatchNorm(Dict(1:2=>normer_p1, 3:4=>normer_p2))
# (;centers, k) = norm
input_depth, selected_outputs = 60, 3:4
x2 = get_x2(y_normed, input_depth, 1:size(y_normed,2), selected_outputs)
x, x2, y_normed_cl = Float32.(x), Float32.(x2), Float32.(y_normed_cl)
data = split_data(x, x2, y_normed_cl, 0.1, false)
(x_train, x_test), (y_train, y_test), (x2_train, x2_test), (p_trn, p_tst) = data

;
#%%

using TensorBoardLogger

lg = TBLogger("tensorboard_logs/run")
!@isdefined(glob_step) && (glob_step = Dict())
glob_step[lg] = 0

# model, opt = create_cluster_model(k)
# train_cl_model!(model, opt, data, 50);
model, opt = create_model(input_depth; selected_outputs=3:4, out_size=50,out_size2=50)

train_model!(model, opt, data, 1, normer_mix);
#%%
train_model!(model, opt, data, 10, normer_mix);
#%%