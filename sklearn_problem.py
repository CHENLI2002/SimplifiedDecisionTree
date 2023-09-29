from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import q7

training_data = q7.Ds
d_32 = training_data[0]
d_128 = training_data[1]
d_512 = training_data[2]
d_2048 = training_data[3]
d_full = training_data[4]
test_set = q7.remaining_test_samples

X_32 = [[item['x1'], item['x2']] for item in d_32]
X_128 = [[item['x1'], item['x2']] for item in d_128]
X_512 = [[item['x1'], item['x2']] for item in d_512]
X_2048 = [[item['x1'], item['x2']] for item in d_2048]
X_full = [[item['x1'], item['x2']] for item in d_full]
X_test = [[item['x1'], item['x2']] for item in test_set]
y_32 = [item['label'] for item in d_32]
y_128 = [item['label'] for item in d_128]
y_512 = [item['label'] for item in d_512]
y_2048 = [item['label'] for item in d_2048]
y_full = [item['label'] for item in d_full]

tree_1 = DecisionTreeClassifier().fit(X_32, y_32)
tree_2 = DecisionTreeClassifier().fit(X_128, y_128)
tree_3 = DecisionTreeClassifier().fit(X_512, y_512)
tree_4 = DecisionTreeClassifier().fit(X_2048, y_2048)
tree_5 = DecisionTreeClassifier().fit(X_full, y_full)

n_32 = tree_1.tree_.node_count
n_128 = tree_2.tree_.node_count
n_512 = tree_3.tree_.node_count
n_2048 = tree_4.tree_.node_count
n_full = tree_5.tree_.node_count

ns = [n_32, n_128, n_512, n_2048, n_full]

errors = [1 - accuracy_score(q7.true_labels, tree_1.predict(X_test)),
         1 - accuracy_score(q7.true_labels, tree_2.predict(X_test)),
         1 - accuracy_score(q7.true_labels, tree_3.predict(X_test)),
         1 - accuracy_score(q7.true_labels, tree_4.predict(X_test)),
         1 - accuracy_score(q7.true_labels, tree_5.predict(X_test))]

for i, n in enumerate(ns):
    print(f" number of nodes of {q7.num_samples[i]} is {n}")

plt.plot(ns, errors)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Learning Curve')
plt.grid(True)
plt.savefig('skLearn n vs err')