import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pprint import pprint


# Variables
global imp
imp = 0.1
max_depth = 10

# Import training set and sort values
df = pd.read_csv('seals_train.csv', sep=' ', header=None)

cols = ['Label']
paramCols = [i for i in range(1, len(df.columns))]
cols.extend(paramCols)

df.columns = cols

labels = df['Label']
df.drop('Label', inplace=True, axis=1)

df['Label'] = labels

# Functions for training tree
# Check purity of node
def check_purity(data):
    entropy, _ = calculate_entropy(data)
    if (len(np.unique(data[:, -1])) < 2) or (entropy <= imp):
        return True
    else:
        return False

# Declare leaf node
def make_leaf(data):
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    majority_index = counts_unique_classes.argmax()
    classification = unique_classes[majority_index]

    return classification

# Check where there is unique values for potential splits
def get_potential_splits(data):
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1):  # excluding the last column which is the label
        potential_splits[column_index] = []
        values = data[:, column_index]
        unique_values = np.unique(values)

        for index in range(len(unique_values)):
            if index != 0:
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].append(potential_split)

    return potential_splits

# Split group to two branches
def split_branch(data, split_column, split_value):
    split_column_values = data[:, split_column]

    left = data[split_column_values <= split_value]
    right = data[split_column_values > split_value]

    return left, right

# Calculate entropy of data
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy, probabilities

# Calculate total entropy of proposed split
def calculate_total_entropy(left, right):
    n = len(left) + len(right)
    p_left = len(left) / n
    p_right = len(right) / n

    left_entropy, left_prob = calculate_entropy(left)
    right_entropy, right_prob = calculate_entropy(right)
    total_entropy = (p_left * left_entropy + p_right * right_entropy)

    return total_entropy, left_prob, right_prob

# Check which of potential split that has lowest total entropy
def determine_best_split(data, potential_splits):
    total_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            left, right = split_branch(data, split_column=column_index, split_value=value)
            current_total_entropy, left_prob, right_prob = calculate_total_entropy(left, right)

            if current_total_entropy <= total_entropy:
                total_entropy = current_total_entropy
                best_split_column = column_index
                best_split_value = value
                left_prob = left_prob
                right_prob = right_prob

    return best_split_column, best_split_value, left_prob, right_prob

# Train/build/grow the tree
def build_decision_tree(df, counter=0, min_samples=2, max_depth=5, threshold=0.5):
    global prob_array
    prob_array = np.array([[0], [0]])
    # Sorting data
    if counter == 0:
        global column_headers
        column_headers = df.columns
        data = df.values
    else:
        data = df

    # Check if leaf
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = make_leaf(data)

        return classification

    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value, left_prob, right_prob = determine_best_split(data, potential_splits)
        left, right = split_branch(data, split_column, split_value)

        # instantiate branches
        feature_name = column_headers[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_branch = {question: []}

        # Find answers (recursion)
        yes_answer = build_decision_tree(left, counter, min_samples, max_depth, threshold)
        no_answer = build_decision_tree(right, counter, min_samples, max_depth, threshold)
        new_prob = np.array([[left_prob], [right_prob]])
        prob_array = np.vstack((prob_array, new_prob))

        # End or split
        if yes_answer == no_answer:
            sub_branch = yes_answer
        else:
            sub_branch[question].append(yes_answer)
            sub_branch[question].append(no_answer)

        return sub_branch


my_test_tree = build_decision_tree(df, max_depth=max_depth, threshold=0.90)
pprint(my_test_tree)

### Classification ###

# Import and sort test set
df_test = pd.read_csv('seals_train.csv', sep=' ', header=None)

cols = ['Label']
paramCols = [i for i in range(1, len(df_test.columns))]
cols.extend(paramCols)

df_test.columns = cols

labels = df_test['Label']
df_test.drop('Label', inplace=True, axis=1)

df_test['Label'] = labels

# Functions for classifying datapoints
def classify_test_set(test_set, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value, _, left_prob, _, right_prob  = question.split()

    # ask question
    if test_set[int(feature_name)] <= float(value):
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer

    # recursive part
    else:
        residual_tree = answer
        return classify_test_set(test_set, residual_tree)


def calculate_accuracy(df_test, tree):
    df_test["classification"] = df_test.apply(classify_test_set, axis=1, args=(tree,))
    df_test["classification_correct"] = df_test["classification"] == df_test["Label"]

    accuracy = df_test["classification_correct"].mean()

    return accuracy


# Make a confusion matrix
def my_confusion_matrix(r_test, test_pred):
    matrix = np.zeros((2, 2)) # initialize the confusion matrix with zeros

    for i in range(2):
        for j in range(2):
            matrix[i, j] = np.sum((r_test == test_pred[i]) & (test_pred == r_test[j]))

    return matrix

# Accuracy from confusion matrix
def confusion_matrix_accuracy(confusion_matrix):
    correct_class = confusion_matrix[0][0] + confusion_matrix[1][1]
    n = correct_class + confusion_matrix[0][1] + confusion_matrix[1][0]
    return correct_class/n

# Create ROC-curve
def roc_curve(df_test, prob):
    labels = df_test['Label'].to_numpy()
    # Creating vektors for TPR and FPR
    FPR_vec = []
    TPR_vec = []
    accuracy = 0
    best_threshold = 0

    for threshold in np.linspace(0, 1, 100):
        pred_labels = []
        for probability in prob:
            if probability[1] > threshold:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        confusion_matrix = my_confusion_matrix(labels, pred_labels)

        tp = confusion_matrix[1][1]
        fp = confusion_matrix[0][1]
        tn = confusion_matrix[0][0]

        TPR = tp / (tp + fp)
        FPR = fp / (fp + tn)  # calculate false positive rate
        if confusion_matrix_accuracy(confusion_matrix) > accuracy:
            accuracy = confusion_matrix_accuracy(confusion_matrix)
            best_threshold = threshold
        TPR_vec.append(TPR)
        FPR_vec.append(FPR)
        pred_labels = []

    auc = roc_auc_score(FPR_vec, TPR_vec)  # calculate area under the ROC curve
    # Print the ROC curve with the AUC value.
    plt.title('ROC')
    plt.plot(FPR_vec, TPR_vec, 'r', label='AUC: %0.3f' % auc)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('2c')
    plt.show()


    return round(best_threshold, 3)



best_threshold = roc_curve(df_test, prob_array)


print("Best calculated threshold is =", best_threshold)


