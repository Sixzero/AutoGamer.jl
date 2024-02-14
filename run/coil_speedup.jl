using Coil, Flux

dense = Dense(3, 6, relu)
compiled_dense = Coil.compile(dense)

x = randn(Float32,3,1);

res = dense(x)
res = compiled_dense(x)

x = randn(Float32, 3, 128)

res = dense(x)
res = compiled_dense(x) # crashes
#%%
import Coil.Tracing
import Coil.IREE
_, tape = Tracing.trace(dense, x; ctx=Tracing.Context(dense));
compiled_tape = Tracing.compile_tape(tape, x; verbose=true, device=IREE.Device("local-task"), hal_target="llvm-cpu", allow_scalar_args=false)

@time res = dense(x)
@show sum(res)
res = compiled_tape(x)
@time res = compiled_tape(x)
@show sum(res)
#%%