
ma_metric(m::Metric, alpha) = mse(m) + alpha * Flux.L2Reg(m)