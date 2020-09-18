import numpy as np
import pandas as pd
import torch


a = torch.randn(3, 3)
print(a)
print(a[0])
b = torch.max(a, 1)
print(b)
b = torch.max(a, 1)[0]
print(b)
b = torch.max(a, 1)[1]
print(b)
print(b.data)
print(b.data.numpy())
c = b.data.numpy()

print(c[0])
a[0, c[0]] = 0
print(a)