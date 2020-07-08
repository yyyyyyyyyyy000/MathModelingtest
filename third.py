import matplotlib.pyplot as plt
from utils import *
import torch
from second import *
import second

top_server_pos_x, top_server_pos_y, main_server_pos_x, main_server_pos_y, sub_server_pos_x, sub_server_pos_y, goods_consume = processData()
goods_resume = torch.tensor([191, 131, 34, 141, 192])
goods_already_consume = torch.zeros_like(goods_resume)
top_distance = get_top_distance()
values, idxs = torch.min(top_distance, dim=0)  # values shape=5
print(second.first_cluster)
print(second.second_cluster)
print(second.third_cluster)
print(second.fourth_cluster)
print(second.fifth_cluster)
second.first_cluster, second.second_cluster, second.third_cluster, second.fourth_cluster, second.fifth_cluster = [], [], [], [], []
# for item in second.first_cluster:
#     goods_resume[0] += goods_consume[item]
# for item in second.second_cluster:
#     goods_resume[1] += goods_consume[item]
# for item in second.third_cluster:
#     goods_resume[2] += goods_consume[item]
# for item in second.fourth_cluster:
#     goods_resume[3] += goods_consume[item]
# for item in second.fifth_cluster:
#     goods_resume[4] += goods_consume[item]
# print(goods_resume)
# print(torch.sum(goods_resume, dim=0))
flag_dict = {'city_0': False,
             'city_1': False,
             'city_2': False,
             'city_3': False,
             'city_4': False}

def reconsider(distance_tensor, top_distance):
    (distance_tensor,_), sub_city_idx, group_idx = getOneItem(distance_tensor)
    if flag_dict['city_' + str(group_idx.item())]:
        flag_dict['city_' + str(group_idx.item())] = False
        distance_tensor[group_idx] -= top_distance
    goods_resume[group_idx] += goods_consume[sub_city_idx]
    return distance_tensor



distance_tensor = initDistance()  # shape=5, 84
for i in range(84):
    (distance_tensor, column_cache), sub_city_idx, group_idx = getOneItem(distance_tensor)
    temp_good_resume = goods_resume[group_idx] + goods_consume[sub_city_idx]
    if temp_good_resume > 2100:
        top_distance = values[group_idx]
        distance_tensor[:, sub_city_idx] = column_cache
        distance_tensor[group_idx] += top_distance
        flag_dict['city_' + str(group_idx.item())] = True
        goods_already_consume[group_idx] += goods_resume[group_idx]
        goods_resume[group_idx] = 0
        distance_tensor = reconsider(distance_tensor, top_distance)
    else:
        if flag_dict['city_' + str(group_idx.item())]:
            flag_dict['city_' + str(group_idx.item())] = False
            distance_tensor[group_idx] -= top_distance
        else:
            goods_resume[group_idx] += goods_consume[sub_city_idx]
for i in range(5):
    goods_already_consume[i] += goods_resume[i]
# print(goods_already_consume)
# print(second.third_cluster)
#
print('*' * 10)
print(second.first_cluster)
print(second.second_cluster)
print(second.third_cluster)
print(second.fourth_cluster)
print(second.fifth_cluster)
def plot():
    plot_basic()
    plot_cluster(second.first_cluster, 0)
    plot_cluster(second.second_cluster, 1)
    plot_cluster(second.third_cluster, 2)
    plot_cluster(second.fourth_cluster, 3)
    plot_cluster(second.fifth_cluster, 4)
    plt.savefig('third.png')
    plt.show()
plot()

