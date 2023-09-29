import matplotlib.pyplot as plt
import main

data = main.get_data("data/q2.txt")
x1 = [i['x1'] for i in data]
x2 = [i['x2'] for i in data]
labels = [i['label'] for i in data]
main.make_sub_tree(data)

colors = ['r' if label == 0 else 'b' for label in labels]

plt.scatter(x1, x2, color=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Training Set')
plt.savefig('q2.png')