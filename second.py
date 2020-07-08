import matplotlib.pyplot as plt
from utils import *
import torch

top_server_pos_x, top_server_pos_y, main_server_pos_x, main_server_pos_y, sub_server_pos_x, sub_server_pos_y = processData()
first_cluster = []
second_cluster = []
third_cluster = []
fourth_cluster = []
fifth_cluster = []


def initDistance():
    return get_distance()
    # return get_total_distance(plot=False)


def updateDistance(distanceTensor: torch.tensor, target_idx, group_idx):
    distanceTensor[:, target_idx] = 10000
    target_x = sub_server_pos_x[target_idx]
    target_y = sub_server_pos_y[target_idx]
    for i in range(84):
        original_distance = distanceTensor[group_idx, i]
        if original_distance == 10000:
            continue
        new_distance = distance((target_x, target_y), (sub_server_pos_x[i], sub_server_pos_y[i]))
        if new_distance < original_distance:
            distanceTensor[group_idx, i] = new_distance
    return distanceTensor


def getOneItem(Distance: torch.tensor):
    rowvalues, rowidx = torch.min(Distance, dim=1)  # size=5
    columnvalues, columnidx = torch.min(rowvalues, dim=0)
    if columnidx == 0:
        first_cluster.append(rowidx[columnidx].item())
    elif columnidx == 1:
        second_cluster.append(rowidx[columnidx].item())
    elif columnidx == 2:
        third_cluster.append(rowidx[columnidx].item())
    elif columnidx == 3:
        fourth_cluster.append(rowidx[columnidx].item())
    else:
        fifth_cluster.append(rowidx[columnidx].item())
    return updateDistance(Distance, rowidx[columnidx], columnidx)


distance_tensor = initDistance()
for i in range(84):
    distance_tensor = getOneItem(distance_tensor)
# print(first_cluster)
# print(second_cluster)
# print(third_cluster)
# print(fourth_cluster)
# print(fifth_cluster)
def plot_cluster(cluster, idx):
    original_point_x = main_server_pos_x[idx]
    original_point_y = main_server_pos_y[idx]
    for i in range(len(cluster)):
        point_x = sub_server_pos_x[cluster[i]]
        point_y = sub_server_pos_y[cluster[i]]
        plt.plot([original_point_x, point_x],[original_point_y, point_y], 'b', alpha=0.2)




plot_basic()
plot_cluster(first_cluster, 0)
plot_cluster(second_cluster, 1)
plot_cluster(third_cluster, 2)
plot_cluster(fourth_cluster, 3)
plot_cluster(fifth_cluster, 4)
plt.savefig('second.png')
plt.show()