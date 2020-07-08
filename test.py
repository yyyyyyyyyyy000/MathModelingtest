import matplotlib.pyplot as plt
import numpy as np
import torch

# plt.scatter([1,2,3], [4,5,6])
# plt.show()
# a = np.array([1])
# print(a.item())
a = torch.randn(5,6)
# a[:,2] = -1
# print(a)
a[4] += 100
print(a)