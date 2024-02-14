module Train
#%%
using Printf
using Flux: train!
using Zygote
using Flux
using CUDA
using Boilerplate
using Random: shuffle

export train_model!
function my_crossentropy(y_pred, Y)
  # Ensure y_pred is within (0, 1) to avoid log(0)
  # y_pred_clipped = clamp.(y_pred, eps(Float32), 1-eps(Float32))
  summing = sum(Y .* log.(y_pred), dims=1)
  # @show summing
  return -mean(summing)
end

loss(m, x, x2, y) = begin
  Y = m(x, x2)
  # is_touched = y[1,:] .>= 0
  # dist_loss = sum(is_touched)>0 ? Flux.mse(Y[1:4,is_touched], y[:,is_touched]) : 0f0
  # true_mask = is_touched
  # notouch_loss = Flux.mse(predicted_mask, true_mask)
  # dist_loss + notouch_loss
  # implement crossentropy
  cord_n_drag_loss = Flux.crossentropy(Y, y)
  # cord_n_drag_loss = my_crossentropy(Y, y)
  cord_n_drag_loss
end
function get_data_selection(x, y, x2, selection)
  x_B, y_B, x2_B = x[:,:,:,selection],y[:,selection],x2[:,selection]
end
accuracy(ŷ, y) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
function train_model!(model, opt, data, epochs=151, normer=nothing)
  (x_train, x_test), (y_train, y_test), (x2_train, x2_test) = data
  parameters_cpu = Flux.params(model)
  m = fmap(cu, model)

  parameters = Flux.params(m)
  @typeof Tuple([p for p in parameters ])
  @sizes Tuple([p for p in parameters ])
  B_full = size(y_train, 2)
  B_full_test = size(y_test, 2)
  B_size = 8
  pool = shuffle(1:B_full)
  epoch, iter, sum_L, sum_D, sum_L_v, sum_D_v = 0, 0, 0., 0., 0., 0. 
  while epoch < epochs
    selection, pool = pool[1:B_size], pool[B_size+1:end]
    x_train_B, y_train_B, x2_train_B = get_data_selection(x_train, y_train, x2_train, selection)
    # only zero by 0.5 chance
    if rand()<0.5
      for i in 0:(size(x2_train_B, 1) ÷ 2)-1
        rand()<0.5 && (x2_train_B[i*2+1:i*2+2, :] .= 0f0)
      end
      # @show x2_train_B[:,1]
    end
    x_train_B_cu, x2_train_B_cu, y_train_B_cu = cu(x_train_B), cu(x2_train_B), cu(y_train_B)
    train!(loss, parameters, [(m, x_train_B_cu, x2_train_B_cu, y_train_B_cu)], opt)
    if length(pool) < B_size
      pool = vcat(pool, shuffle(1:B_full))
      iter, epoch = 0, epoch + 1
      @show sum_L, sum_D, sum_L_v, sum_D_v
      sum_L, sum_D, sum_L_v, sum_D_v = 0., 0., 0., 0.
    end
    if iter % 10 === 0
      # non_zeros = findall(v-> v!=1, y_train[1,:])
      idx_train_valid, idx_test_valid = iter % (B_full-20), iter % (B_full_test-20)
      sel_train, sel_test = shuffle(1:size(x_train,4))[1:20], shuffle(1:size(x_test,4))[1:20]
      sel_train, sel_test = idx_train_valid+1:idx_train_valid+20, idx_test_valid+1:idx_test_valid+20
      x_train_B, y_train_B, x2_train_B = get_data_selection(x_train, y_train, x2_train, sel_train)
      x_test_B, y_test_B, x2_test_B = get_data_selection(x_test, y_test, x2_test, sel_test)
      x_train_B_cu, y_train_B_cu, x2_train_B_cu, x_test_B_cu, y_test_B_cu, x2_test_B_cu = cu(x_train_B), cu(y_train_B), cu(x2_train_B), cu(x_test_B), cu(y_test_B), cu(x2_test_B)
      res_train, res_test = m(x_train_B_cu, x2_train_B_cu),m(x_test_B_cu, x2_test_B_cu)
      l1, l2 = loss(m, x_train_B_cu, x2_train_B_cu, y_train_B_cu), loss(m, x_test_B_cu, x2_test_B_cu, y_test_B_cu)
      acc1, acc2 = accuracy(res_train, y_train_B_cu), accuracy(res_test, y_test_B_cu)
      coord1, coord2 = denorm(normer, cpu(res_train)), denorm(normer, y_train_B)
      coord1_v, coord2_v = denorm(normer, cpu(res_test)), denorm(normer, y_test_B)
      dist, dist_v = Flux.mse(coord1[1:2,:], coord2[1:2,:]), Flux.mse(coord1_v[1:2,:], coord2_v[1:2,:])
      sum_L, sum_D, sum_D_v, sum_L_v = sum_L + l1, sum_D + dist, sum_D_v + dist_v, sum_L_v + l2

      log_value(lg, "test/loss", l1, step=glob_step[lg]) 
      log_value(lg, "test/loss_v", l2, step=glob_step[lg]) 
      log_value(lg, "test/dist", dist, step=glob_step[lg]) 
      log_value(lg, "test/dist_v", dist_v, step=glob_step[lg]) 
      @printf("%3d./%3d. T.: L: %.4f %.4f D: %.4f %.4f acc: %.4f %.4f\n", epoch, iter, l1, l2, dist, dist_v, acc1, acc2)
    end
    glob_step[lg] += 1
    iter += B_size
    # iter > 1200 && break
  end
  # @show accuracy(m(x_test, x2_test), y_test)
  # assign params back from GPU to CPU model (m)  
  for (i, p) in enumerate(parameters)
    parameters_cpu[i] .= Array(p)
  end
end
function train_cl_model!(model, opt, data, epochs=151)
  loss(x, y) = Flux.crossentropy(model(x), y)
  
  (x_train, x_test), (y_train, y_test)  = data
  parameters = Flux.params(model)
  @show length(parameters)
  B_full = size(y_train, 2)
  B_size = 4
  pool = 1:B_full
  epoch = 0
  while epoch < epochs
    selection, pool = pool[1:B_size], pool[B_size+1:end]
    x_train_B, y_train_B = x_train[:,:,:,selection],y_train[:,selection]
    train!(loss, parameters, [(x_train_B, y_train_B)], opt)
    if length(pool) < B_size
      if epoch % 1 === 0
        res_train = model(x_train)
        res_test = model(x_test)
        @printf("%3d. T.: mse: %.4f %.4f cross: %.4f %.4f acc: %.4f %.4f\n", epoch, 
        Flux.mse(res_train, y_train), Flux.mse(res_test, y_test),

      Flux.crossentropy(res_train, y_train), Flux.crossentropy(res_test, y_test),
                      accuracy(res_train, y_train), accuracy(res_test, y_test))
      end
      pool = vcat(pool, shuffle(1:B_full))
      epoch += 1
    end
  end
  @show accuracy(model(x_test), y_test)
  @show Flux.onecold(model(x_test))
end

#%%
using Plots
# using ImageView, Images
plot_wrong_predictions(model, x_train, y_train) = begin
  indices_of_wrong(ŷ, y) = Flux.onecold(ŷ) .== Flux.onecold(y)
  misses = x_train[:,:,:,.!indices_of_wrong(model(x_train), y_train)]
  @sizes misses
  for i in 1:size(misses, 4)
    x = misses[:,:,:,i]
    m = RGB{N0f8}.(x, x, x)
    imshow(m)
  end
  
end

onecold(y) = Flux.onecold(y)
#%%

end