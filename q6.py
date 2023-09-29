import matplotlib.pyplot as plt
import numpy as np
import main
import q5
import random

data_d1 = np.loadtxt("data/D1.txt")
data_d2 = np.loadtxt("data/D2.txt")


#print(data_d1)
x1_d1 = [item[0] for item in data_d1]
x2_d1 = [item[1] for item in data_d1]
y_d1 = [main.predict(q5.tree_1, {'x1':item[0], 'x2':item[1], 'y':item[2]}) for item in data_d1]
x1_d2 = [item[0] for item in data_d2]
x2_d2 = [item[1] for item in data_d2]
y_d2 = [main.predict(q5.tree_2, {'x1':item[0], 'x2':item[1], 'y':item[2]}) for item in data_d1]

colour_d1 = ['r' if y == 0 else 'b' for y in y_d1]
colour_d2 = ['b' if y == 0 else 'g' for y in y_d2]
plt.subplot(2, 1, 1)
plt.scatter(x1_d1,x2_d1,color=colour_d1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("D1 plot")
plt.subplot(2, 1, 2)
plt.scatter(x1_d2, x2_d2, color=colour_d2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("D2 plot")
plt.savefig('Q6 ans')

