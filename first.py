import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import math
from matplotlib.patches import Circle
from utils import *


top_server_pos_x, top_server_pos_y, main_server_pos_x, main_server_pos_y, sub_server_pos_x, sub_server_pos_y, _ = processData()
total_distance: torch.tensor = get_total_distance()
top_distance = get_top_distance()
values, idxs = torch.min(total_distance, dim=0)
first = []
second = []
third = []
fourth = []
fifth = []
for i in range(values.size(0)):
    if idxs[i] == 0:
        first.append(values[i])
    elif idxs[i] == 1:
        second.append(values[i])
    elif idxs[i] == 2:
        third.append(values[i])
    elif idxs[i] == 3:
        fourth.append(values[i])
    else:
        fifth.append(values[i])

base_value = torch.min(top_distance, dim=0)[0]

max_r1, _ = torch.max(torch.tensor(first), dim=0)
max_r1 -= base_value[0]
max_r2, _ = torch.max(torch.tensor(second), dim=0)
max_r2 -= base_value[1]
max_r3, _ = torch.max(torch.tensor(third), dim=0)
max_r3 -= base_value[2]
max_r4, _ = torch.max(torch.tensor(fourth), dim=0)
max_r4 -= base_value[3]
max_r5, _ = torch.max(torch.tensor(fifth), dim=0)
max_r5 -= base_value[4]
plot_basic()
circle1 = plt.Circle(xy=(main_server_pos_x[0], main_server_pos_y[0]), alpha=0.3, radius=max_r1.item(), color='b')
circle2 = plt.Circle(xy=(main_server_pos_x[1], main_server_pos_y[1]), alpha=0.3, radius=max_r2.item(), color='b')
circle3 = plt.Circle(xy=(main_server_pos_x[2], main_server_pos_y[2]), alpha=0.3, radius=max_r3.item(), color='b')
circle4 = plt.Circle(xy=(main_server_pos_x[3], main_server_pos_y[3]), alpha=0.3, radius=max_r4.item(), color='b')
circle5 = plt.Circle(xy=(main_server_pos_x[4], main_server_pos_y[4]), alpha=0.3, radius=max_r5.item(), color='b')
plt.gcf().gca().add_artist(circle1)
plt.gcf().gca().add_artist(circle2)
plt.gcf().gca().add_artist(circle3)
plt.gcf().gca().add_artist(circle4)
plt.gcf().gca().add_artist(circle5)
plt.savefig('first.png')
plt.show()
