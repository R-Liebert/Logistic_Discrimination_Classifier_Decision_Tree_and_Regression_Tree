import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# Selfmade-ish Functions
def error(r, y):
    return -np.sum(r*np.log(y+0.001)+(1-r)*np.log(1-y+0.001))


def sigmoid(X, w):
    return 1 / (1 + np.exp(-np.matmul(X, w)))


def train_model(true_label, X_train, weights, eta, iteration=0, e_old=1, e_new=0, y=0):
    while abs(e_old - e_new) > 0.1:
        iteration += 1
        e_old = (1/10)*error(true_label[iteration], y)
        y = sigmoid(X_train, weights)
        delta_w = np.matmul(X_train.T, true_label-y.T)
        weights = weights + eta*delta_w
        e_new = (1/10)*error(r_train[iteration], y)
    return weights, iteration


def accuracy(w, x, r, threshold=0.5):
    predicted_classes = (sigmoid(x, w) >=
                         threshold).astype(int)
    accuracy = np.mean(predicted_classes == r)
    return accuracy * 100


def my_confusion_matrix(r_test, pred):
    matrix = np.zeros((2, 2)) # initialize the confusion matrix with zeros

    for i in range(2):
        for j in range(2):
            matrix[i, j] = np.sum((r_test == pred[i]) & (pred == r_test[j]))

    return matrix


def roc_curve(x_test, r_test, w):
    thresholds = np.linspace(0, 1, 1000)
    True_positive_rate_vec = np.zeros_like(thresholds)
    False_positive_rate_vec = np.zeros_like(thresholds)

    # Make a loop for making thresholds
    for j in range(999):
        # Classify test set according to weights and create (near) empty confusion matrix
        o_roc = sigmoid(x_test, w)
        confusion_matrix_roc = [[0.0, 0.01],
                                [0.0, 0.01]]

        for i in range(len(r_test)):
            if o_roc[i] > j/998:
                o_roc[i] = 1
            else:
                o_roc[i] = 0

            # Use my confusion matrix function for making confusion matrix with current threshold
            confusion_matrix_roc = my_confusion_matrix(r_test, o_roc)

        # Create variables for True Positive, False Positive, True Negative, False Negative
        tp_roc = confusion_matrix_roc[1][1]
        fp_roc = confusion_matrix_roc[0][1]
        tn_roc = confusion_matrix_roc[0][0]
        fn_roc = confusion_matrix_roc[1][0]

        # Add new rates to array
        True_positive_rate_vec[j] = (tp_roc)/(tp_roc + fn_roc)
        False_positive_rate_vec[j] = (fp_roc)/(fp_roc + tn_roc)

    return True_positive_rate_vec, False_positive_rate_vec


# Task 1b

# Variables
eta = 0.01
convergence = 1
y = 0
threshold = 0.5


# Import training set
df_train = pd.read_csv('seals_train.csv', sep=' ', header=None)

cols = ['Label']
paramCols = [i for i in range(1, len(df_train.columns))]
cols.extend(paramCols)

df_train.columns = cols

# Create arrays for training
X_train = df_train.drop('Label', axis=1).to_numpy() # Create array of parameters
X_train = np.insert(X_train, 0, 1.0, axis=1) # Insert ones in first column
X_train = X_train.astype('float64')

r_train = df_train['Label'].to_numpy() # Create array of ground truth

# Import test set
df_test = pd.read_csv('seals_test.csv', sep=' ', header=None)

df_test.columns = cols

# Create arrays for test
X_test = df_test.drop('Label', axis=1).to_numpy()  # Create array of parameters
X_test = np.insert(X_test, 0, 1.0, axis=1)  # Insert ones in first column

r_test = df_test['Label'].to_numpy()  # Create arrays of ground truth

# Vector for initial weights
w = np.random.uniform(-0.01, 0.01, np.shape(X_train)[1])


# Train weights
w, _ = train_model(r_train, X_train, w, eta)


# Predicted result
o = sigmoid(X_test, w)

# assign to class depending of threshold
for t in range(len(o)):
    if o[t] > threshold:
        o[t] = 1
    else:
        o[t] = 0


# Check accuracy and confusion matrix of test
acc_test = accuracy(w, X_test, r_test, threshold)

confusion_matrix = pd.crosstab(r_test, o, rownames=['Actual'], colnames=['Predicted'])

print("Confusion Matrix for seal test set: \n")
print(confusion_matrix)
print("\nAccuracy on test set: %.3f \n" % acc_test)


# Task 1c
# Calculate AUC
auc = roc_auc_score(r_test, o)
print('AUC: %.3f' % auc)


# Create TPR- and FPR-arrays for ROC curve
Tpr_vec, Fpr_vec = roc_curve(X_test, r_test, w)

# plot the roc curve for the model
plt.plot(Fpr_vec, Tpr_vec, 'b', label='AUC: %.3f' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-curve and AUC for seal test set')
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()

# Task 1d

# Print correct and wrong classified images
# make list of correct and false classified seals
tp = confusion_matrix[1][1]
fp = confusion_matrix[0][1]
tn = confusion_matrix[0][0]
fn = confusion_matrix[1][0]


# List of correct and false classified seals
right_seal = np.zeros((tp+tn), dtype='int')
wrong_seal = np.zeros((fp+fn), dtype='int')
t = 0  # Index for list of correct class
f = 0  # Index for list of wrong class


for index in range(len(r_test)):
    if r_test[index] == o[index]:
        right_seal[t] = index
        t += 1
    else:
        wrong_seal[f] = index
        f += 1

# Load the image file
df_test_img = pd.read_csv("seals_images_test.csv", sep=' ', header=None)
images = df_test_img.to_numpy()

# Extract the first five correctly classified seals
for i in range(5):
    one_right_image = images[right_seal[i]]

    # Reshape it to be 64 x 64
    reshaped_right_image = np.reshape(one_right_image, (64, 64))

    # show the image
    plt.imshow(reshaped_right_image, cmap='gray')
    plt.title('Correct classified image no. {}'.format(i+1))
    plt.show()

# Extract the first five misclassified seals
for i in range(5):
    one_wrong_image = images[wrong_seal[i]]

    # Reshape it to be 64 x 64
    reshaped_wrong_image = np.reshape(one_wrong_image, (64, 64))

    # show the image
    plt.imshow(reshaped_wrong_image, cmap='gray')
    plt.title('False classified image no. {}'.format(i+1))
    plt.show()
