
#%%
using Plots
idk = state.resolution .* (normer_p2.centers)
Plots.plot(idk[1,:], idk[2,:], seriestype=:scatter)
Plots.plot!([-50], [64], color=:red, seriestype=:scatter)
@show idk

#%%
using CUDA
model = fmap(cpu, model)
#%%
#%%
using CUDA
# model, opt = create_model(input_depth; input_dim=4)
m = fmap(cu, model)
x_train_B, y_train_B, x2_train_B = get_data_selection(x_train, y_train, x2_train, 1:10)
@sizes x_train_B
@typeof m
parameters = Flux.params(m)
@typeof Tuple([p for p in parameters ])
@sizes Tuple([p for p in parameters ])
# m(x_train_B, x2_train_B)
#%%
@time model(x_train_B, x2_train_B)
@time model(x_train_B, x2_train_B)
@time m(cu(x_train_B), cu(x2_train_B))
@time m(cu(x_train_B), cu(x2_train_B))

#%%
create_model(input_depth)
#%%

y_pred_trn = denorm(model(x_train), normer)
y_pred_tst = denorm(model(x_test), normer)

diff_trn = abs.(y_pred_trn .- y[:, p_trn])
diff_tst = abs.(y_pred_tst .- y[:, p_tst])

@show mean(diff_trn)
@show mean(diff_tst)
@show mean(diff_trn)
@show maximum(diff_tst, dims=2)
# @show the value where the "diff_tst" is maximum
max_diff_idx = argmax(diff_tst)
@show max_diff_idx
@show diff_tst[:, max_diff_idx[2]] ./ normer.mean
@show y[:, p_tst][:,max_diff_idx[2]]
@show y_pred_tst[:,max_diff_idx[2]]
#%%

#%%
using TensorBoardLogger

lg=TBLogger("tensorboard_logs/run3")
#%%
  for i=1:100
      log_value(lg, "test/i", 1.5*i, step=i) 
  end
#%%
run(`rm -rf tensorboard_logs/run`)
#%%
plot_convert_coords(y_normed[3:4,:], 10:100)
#%%
c = kmeans(y_normed[3:4,:], 50).centers
plot(c[1,:], c[2,:], seriestype=:scatter)
#%%
plot_convert_coords(y_normed[1:2,:], 10:100)

#%%
plot(y_normed[3,:], y_normed[4,:], seriestype=:scatter)
sum(y_normed[3,:] .== 0 .&& y_normed[4,:] .== 0)
# sum(y[1,:] .== -1 .&& y[2,:] .== -1)
# y_normed[1,y[1,:] .== -1]
#%%
# get armax on each row
@show argmax(diff_tst, dims=2)
#%%
Flux.mse(denorm(model(x_train), centers), y[:, p_trn])
#%%

(x_train, x_test), (y_train, y_test) = data
res_train = model(x_train)
res_test = model(x_test)
diff = res_train .- y_train
# display diff with 3 digit precision
@display round.(diff, digits=2)
@display round.(res_test .- y_test, digits=2)
@display y_test