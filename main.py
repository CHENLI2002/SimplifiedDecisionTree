import math
import numpy as np

class Node:
    def __init__(self):
        self.split = None
        self.feature = None
        self.children = {}
        self.label = None

def make_sub_tree(data):
    count_0, count_1 = count_label(data)

    candidates, entropy, ratio = find_candidates(data, count_0, count_1)

   #print(candidates)

    if not entropy or ratio == 0 or len(data) == 0 or len(candidates) == 0:

        leaf_node = Node()

        if count_0 == count_1:
            leaf_node.label = 1
        elif count_0 > count_1:
            leaf_node.label = 0
        else:
            leaf_node.label = 1

        return leaf_node
    else:
        node = Node()
        best, feature = find_best(data, candidates)
        node.split = best
        node.feature = feature

        node.children['then'] = make_sub_tree([item for item in data if item[node.feature] >= node.split])
        node.children['else'] = make_sub_tree([item for item in data if item[node.feature] < node.split])

        return node

def find_candidates(data, count_0, count_1):
    sorted_data_x1 = sorted(data, key=lambda x:x['x1'])
    sorted_data_x2 = sorted(data, key=lambda x:x['x2'])
    entropy = False

    collection = []
    sum_of_ratio = 0

    prev = -1

    for index, entry in enumerate(sorted_data_x1):
        if prev == -1:
            prev = entry['label']
            continue

        if prev != entry['label']:
            prev = entry['label']
            split = entry['x1']
            left_data = [item for item in data if item['x1'] >= split]
            right_data = [item for item in data if item['x1'] < split]
            entropy = get_entropy(len(left_data), len(right_data))

            if entropy == 0:
                H_D_Y = get_entropy(count_0, count_1)
                count_0_left, count_1_left = count_label(left_data)
                count_0_right, count_1_right = count_label(right_data)
                H_D_Y_left = get_entropy(count_0_left, count_1_left)
                H_D_Y_right = get_entropy(count_0_right, count_1_right)
                info_gain = H_D_Y - (len(left_data)/len(data)*H_D_Y_left + len(right_data)/len(data)*H_D_Y_right)
                #print(f"split using x1, split at {split}, entropy is zero but infoGain is {info_gain}")
                continue
            else:
                entropy = True
                H_D_Y = get_entropy(count_0, count_1)
                count_0_left, count_1_left = count_label(left_data)
                count_0_right, count_1_right = count_label(right_data)
                H_D_Y_left = get_entropy(count_0_left, count_1_left)
                H_D_Y_right = get_entropy(count_0_right, count_1_right)
                info_gain = H_D_Y - (len(left_data)/len(data)*H_D_Y_left + len(right_data)/len(data)*H_D_Y_right)
                gain_rate = info_gain / entropy
                sum_of_ratio += abs(gain_rate)
                collection.append(['x1', split, gain_rate])

    prev = -1

    for index, entry in enumerate(sorted_data_x2):
        if prev == -1:
            prev = entry['label']
            continue

        if prev != entry['label']:
            prev = entry['label']
            split = entry['x2']
            left_data = [item for item in data if item['x2'] >= split]
            right_data = [item for item in data if item['x2'] < split]
            entropy = get_entropy(len(left_data), len(right_data))

            if entropy == 0:
                H_D_Y = get_entropy(count_0, count_1)
                count_0_left, count_1_left = count_label(left_data)
                count_0_right, count_1_right = count_label(right_data)
                H_D_Y_left = get_entropy(count_0_left, count_1_left)
                H_D_Y_right = get_entropy(count_0_right, count_1_right)
                info_gain = H_D_Y - (len(left_data)/len(data)*H_D_Y_left + len(right_data)/len(data)*H_D_Y_right)
                #print(f"split using x2, split at {split}, entropy is zero but infoGain is {info_gain}")
                continue
            else:
                entropy = True
                H_D_Y = get_entropy(count_0, count_1)
                count_0_left, count_1_left = count_label(left_data)
                count_0_right, count_1_right = count_label(right_data)
                H_D_Y_left = get_entropy(count_0_left, count_1_left)
                H_D_Y_right = get_entropy(count_0_right, count_1_right)
                info_gain = H_D_Y - (len(left_data)/len(data)*H_D_Y_left + len(right_data)/len(data)*H_D_Y_right)
                gain_rate = info_gain / entropy
                sum_of_ratio += abs(gain_rate)
                collection.append(['x2', split, gain_rate])

    return collection, entropy, sum_of_ratio

def count_label(data):
    count_0 = 0
    count_1 = 0

    for entry in data:
        if entry['label'] == 0:
            count_0 += 1
        else:
            count_1 += 1

    return count_0, count_1

def get_entropy(left, right):
    sum = left + right

    if left == 0:
        left_term = 0
    else:
        left_term = (left/sum) * math.log2(left/sum)
    if right == 0:
        right_term = 0
    else:
        right_term = (right/sum) * math.log2(right/sum)

    return -(left_term+right_term)

def find_best(data, candidates):
    sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return sorted_candidates[0][1], sorted_candidates[0][0]

def get_data(file):
    data = np.loadtxt(file)
    data_as_dict = [{'x1': float(row[0]), 'x2': float(row[1]), 'label': int(row[2])} for row in data]
    return data_as_dict

def predict(node, data):
    if node.label != None:
        return node.label
    else:
        if data[node.feature] >= node.split:
            return predict(node.children['then'], data)
        else:
            return predict(node.children['else'], data)

def count_nodes(tree):
    if tree.label != None:
        return 1
    else:
        return 1 + count_nodes(tree.children['then']) + count_nodes(tree.children['else'])


