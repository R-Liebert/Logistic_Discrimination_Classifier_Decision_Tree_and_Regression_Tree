import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('uncensored_data.csv', sep=' ', header=None, names=['0', '1', '2'])
df_cens = pd.read_csv('censored_data.csv', sep=' ', header=None, names=['0', '1', '2'])
df_cens = df_cens.dropna()

plt.scatter(df_cens['0'], df_cens['1'])
plt.plot([0, 10], [-4.947, 4.993], 'k-', color='r')
plt.title('Linear regression on column 0 and 1 in censored_data.csv')
plt.xlabel('0')
plt.ylabel('1')
plt.show()

threedeebiaatch = plt.figure().gca(projection='3d')
threedeebiaatch.scatter(df['0'], df['1'], df['2'])
threedeebiaatch.set_xlabel('0')
threedeebiaatch.set_ylabel('1')
threedeebiaatch.set_zlabel('2')
threedeebiaatch.set_title('Plot of datapoints in uncensored_data.csv')
plt.show()


plt.scatter(df['0'], df['1'], color='y')
plt.plot([0, 5.90], [-2.43, -2.43], 'k-', color='r')
plt.plot([5.90, 10.1], [2.7, 2.7], 'k-', color='r')

plt.plot([0, 1.38], [-4.53, -4.53], 'k-', color='g')
plt.plot([1.38, 5.09], [-1.64, -1.64], 'k-', color='g')
plt.plot([5.09, 7.255], [0.97, 0.97], 'k-', color='g')
plt.plot([7.255, 10.1], [3.957, 3.957], 'k-', color='g')


plt.xlabel('0')
plt.ylabel('1')
plt.title('Plot of datapoints in column 0 and 1 from uncensored_data.csv')
plt.show()
total_accuracy = [1.6918, 1.6918, 1.6918, 1.6918, 1.6918, 1.6918, 1.6918, 1.6918, 1.5957, 1.5178, 1.5178]
tree_max_depth_total = [1.5178, 1.6537, 1.6918, 1.7531, 1.7746, 1.8028, 1.8774, 1.9271, 1.9718, 1.9917]
tree_max_depth_test = [0.7589, 0.8268, 0.8459, 0.8766, 0.8873, 0.9014, 0.9387, 0.9635, 0.9859, 0.9959]
x = np.linspace(1, 10, 10)



plt.plot(x, total_accuracy)
plt.title("Combined accuracy of training and test set on tree with max depth of 3")
plt.ylabel("Total accuracy")
plt.xlabel("Minimum impurity")
plt.show()

plt.plot(x, tree_max_depth_total, label='Total accuracy')
plt.plot(x, tree_max_depth_test, label='Test accuracy')
plt.title("Accuracy of dataset on tree with max impurity of 0.1")
plt.ylabel("Accuracy")
plt.grid()
plt.xlabel("Max depth of tree")
plt.show()