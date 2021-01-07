import pandas as pd
import numpy as np
from pprint import pprint

# Variabler
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
    if (len(np.unique(data[:, -1])) < 2) or (calculate_entropy(data) <= imp):
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
    for column_index in range(n_columns - 1):  # Stop before column with label
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

# Split branch when we have determined best split
def split_branch(data, split_column, split_value):
    split_column_values = data[:, split_column]

    left = data[split_column_values <= split_value]
    right = data[split_column_values > split_value]

    return left, right

# Calculate the entropy of group of data
def calculate_entropy(data):
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))

    return entropy

# Calculate total entropy of both branches
def calculate_total_entropy(left, right):
    n = len(left) + len(right)
    p_left = len(left) / n
    p_right = len(right) / n

    total_entropy = (p_left * calculate_entropy(left)
                     + p_right * calculate_entropy(right))

    return total_entropy

# Find the splits with lowest total entropy
def determine_best_split(data, potential_splits):
    # Start with an impossible high value that calc entropy won't be higher than
    first_iter = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            left, right = split_branch(data, split_column=column_index, split_value=value)
            current_total_entropy = calculate_total_entropy(left, right)

            if first_iter == True or current_total_entropy <= total_entropy:
                first_iter = False

                total_entropy = current_total_entropy
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value

# Train/build/grow tree on the trainingset
def build_decision_tree(df, counter=0, min_samples=2, max_depth=5):
    # Sorting data
    if counter == 0:
        global column_headers
        column_headers = df.columns
        data = df.values
    else:
        data = df

    # For task 2b
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = make_leaf(data)

        return classification


    # recursive part
    else:
        counter += 1

        # helper functions
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        left, right = split_branch(data, split_column, split_value)

        # instantiate sub-tree
        feature_name = column_headers[split_column]
        question = "{} <= {}".format(feature_name, split_value)
        sub_branch = {question: []}

        # Find answers (recursion)
        yes_answer = build_decision_tree(left, counter, min_samples, max_depth)
        no_answer = build_decision_tree(right, counter, min_samples, max_depth)


        # Check if we're at the end or append split
        if yes_answer == no_answer:
            sub_branch = yes_answer
        else:
            sub_branch[question].append(yes_answer)
            sub_branch[question].append(no_answer)

        return sub_branch

# build tree with the training set
my_tree = build_decision_tree(df, max_depth=max_depth)


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
    feature_name, comparison_operator, value = question.split()

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


# Check accuracy of decision tree
accuracy_test = calculate_accuracy(df_test, my_tree)
print("Accuracy of decision tree on test set = %.3f." % accuracy_test)

# Some tests for determining depth and threshold for purity
#accuracy_train = calculate_accuracy(df, my_tree)
#total_accuracy = accuracy_test + accuracy_train


#confusion_matrix = pd.crosstab(df_test["Label"], df_test["classification"], rownames=['Actual'], colnames=['Predicted'])

#print("Confusion Matrix for seal test set: \n")
#print(confusion_matrix)
#print("\nAccuracy on test set: %.3f \n" % accuracy_test)


