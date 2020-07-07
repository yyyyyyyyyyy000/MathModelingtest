import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import math
from matplotlib.patches import Circle
fi_name = 'data.xlsx'
df = pd.read_excel(fi_name)
city_no = df.loc[:, '城市序号'].values
city_pos_x = df.loc[:, 'X坐标'].values
city_pos_y = df.loc[:, 'Y坐标'].values
city_type = df.loc[:, '类型'].values
city_goods = df.loc[:, '商品需求量'].values

# plt.scatter(city_pos_x, city_pos_y)
# plt.show()
sub_server_pos_x = []
sub_server_pos_y = []
main_server_pos_x = []
main_server_pos_y = []
top_server_pos_x = []
top_server_pos_y = []


def distance(first, second):
    first_x = first[0]
    first_y = first[1]
    second_x = second[0]
    second_y = second[1]
    distance = math.sqrt((first_x - second_x) ** 2 + (first_y - second_y) ** 2)
    return distance


for i in range(city_no.shape[0]):
    if city_type[i] == '一级代理':
        main_server_pos_x.append(city_pos_x[i])
        main_server_pos_y.append(city_pos_y[i])
    elif city_type[i] == '分销商':
        sub_server_pos_x.append(city_pos_x[i])
        sub_server_pos_y.append(city_pos_y[i])
    else:
        top_server_pos_x.append(city_pos_x[i])
        top_server_pos_y.append(city_pos_y[i])



def get_top_distance():
    top_distance = torch.zeros(2, 5)
    for i in range(2):
        for j in range(5):
            first_point = (top_server_pos_x[i], top_server_pos_y[i])
            second_point = (main_server_pos_x[j], main_server_pos_y[j])
            single_distance = distance(first_point, second_point)
            top_distance[i, j] = single_distance
    return top_distance


def get_distance():
    distance_tensor = torch.zeros(5, 84)
    for i in range(5):
        for j in range(84):
            first_point = (main_server_pos_x[i], main_server_pos_y[i])
            second_point = (sub_server_pos_x[j], sub_server_pos_y[j])
            single_distance = distance(first_point, second_point)
            distance_tensor[i, j] = single_distance
    return distance_tensor

def plot_line(idx):
    for i in range(idx.size(0)):
        id = idx[i]
        plt.plot([top_server_pos_x[id], main_server_pos_x[i]],[top_server_pos_y[id], main_server_pos_y[i]], 'y')

def get_total_distance():
    top_main_distance = get_top_distance()
    main_sub_distance = get_distance()
    value, idx = torch.min(top_main_distance, dim=0)
    plot_line(idx)
    # print(top_main_distance)
    # print(value)
    # print(idx)
    total_distance = main_sub_distance + value.resize_(5, 1)
    return total_distance


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
b = plt.scatter(main_server_pos_x, main_server_pos_y, c='r')
c = plt.scatter(sub_server_pos_x, sub_server_pos_y, c='g')
a = plt.scatter(top_server_pos_x, top_server_pos_y, c='b')
plt.legend([a,b,c],['top_city', 'main_city', 'sub_city'])
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
