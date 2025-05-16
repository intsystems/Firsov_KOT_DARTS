import time
import torch
import torch.nn as nn
import numpy as np
import json
from models.cnn import ops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dummy_input = torch.randn(16, 32, 32, 32).to(device)

primitives = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 
              'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5']

latency_results = {}
num_warmup = 10000
num_iters = 100000

for prim in primitives:
    op = ops.OPS[prim](32, 1, affine=False).to(device)
    op.eval()

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = op(dummy_input)
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            start = time.time()
            _ = op(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    avg_time = np.mean(times)
    latency_results[prim] = avg_time

with open("latency_table.json", "w") as f:
    json.dump(latency_results, f, indent=4)

print("Measured latency (s):")
for k, v in latency_results.items():
    print(f"{k}: {v:.6f}")
