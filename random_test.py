import random
import numpy as np
import time

N = 10000000

dur_np = []
for _ in range(N):
    start = time.perf_counter()
    a = np.random.rand()
    dur_np.append(time.perf_counter() - start)
print(f"{np.mean(dur_np) = }")


dur_std = []
for _ in range(N):
    start = time.perf_counter()
    a = random.random()
    dur_std.append(time.perf_counter() - start)
print(f"{np.mean(dur_std) = }")