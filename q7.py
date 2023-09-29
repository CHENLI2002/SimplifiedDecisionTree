import main
import random
import numpy as np
import matplotlib.pyplot as plt

data = main.get_data("data/Dbig.txt")
random_sample = random.sample(data, 8192)
remaining_test_samples = [item for item in data if item not in random_sample]
#print(random_sample)
D32 = random_sample[slice(32)]
D128 = random_sample[slice(128)]
D512 = random_sample[slice(512)]
D2048 = random_sample[slice(2048)]
D8192 = random_sample

Ds = [D32, D128, D512, D2048, D8192]

tree_32 = main.make_sub_tree(D32)
tree_128 = main.make_sub_tree(D128)
tree_512 = main.make_sub_tree(D512)
tree_2048 = main.make_sub_tree(D2048)
tree_8192 = main.make_sub_tree(D8192)

n = [main.count_nodes(tree_32), main.count_nodes(tree_128), main.count_nodes(tree_512), main.count_nodes(tree_2048),
     main.count_nodes(tree_8192)]

predictions = [[main.predict(tree_32, {'x1':item['x1'], 'x2':item['x2'], 'y':item['label']}) for item in remaining_test_samples],
               [main.predict(tree_128, {'x1':item['x1'], 'x2':item['x2'], 'y':item['label']}) for item in remaining_test_samples],
               [main.predict(tree_512, {'x1':item['x1'], 'x2':item['x2'], 'y':item['label']}) for item in remaining_test_samples],
               [main.predict(tree_2048, {'x1':item['x1'], 'x2':item['x2'], 'y':item['label']}) for item in remaining_test_samples],
               [main.predict(tree_8192, {'x1':item['x1'], 'x2':item['x2'], 'y':item['label']}) for item in remaining_test_samples]]

true_labels = [item['label']for item in remaining_test_samples]

errors = []

for prediction in predictions:
     nPre = np.array(prediction)
     nTrue = np.array(true_labels)
     error = np.sum(nPre != nTrue)
     errors.append(error / len(true_labels))

for nodes, err in zip(n, errors):
    print(f'n={nodes}, err={err}')

plt.plot(n, errors)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.grid(True)
plt.savefig('Q7 n vs err')
plt.clf()

x1_test = [item['x1'] for item in remaining_test_samples]
x2_test = [item['x2'] for item in remaining_test_samples]

num_samples = [32, 128, 512, 2048, 8192]

for index, prediction in enumerate(predictions):
    colour = ['r' if y == 0 else 'b' for y in prediction]
    plt.scatter(x1_test, x2_test, c=colour)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f"{num_samples[index]} prediction visualization")
    plt.savefig(f"{num_samples[index]} prediction visualization")

plt.clf()