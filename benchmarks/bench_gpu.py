import time, symbiote.gpu_backend as gpu

sim = gpu.Simulation(4096,4096)
t0 = time.perf_counter()
for _ in range(128): sim.step()
print('Elapsed', time.perf_counter()-t0)
