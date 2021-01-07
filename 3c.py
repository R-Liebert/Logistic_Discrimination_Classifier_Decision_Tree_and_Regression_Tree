import pandas as pd
import numpy as np
from pprint import pprint


# Variables
max_depth = 3

# Import training set and sort values
df_censored_train = pd.read_csv('censored_data.csv', sep=' ', header=None)
df_none_nan = df_censored_train.dropna()
cols = df_none_nan.columns.tolist()
cols = [0, 2, 1]
df_none_nan = df_none_nan[cols]

# Functions for training tree
# Check purity of node
def check_purity(data):
    label_column = data[:-1]
    unique_values = np.unique(label_column)

    if len(unique_values) == 1:
        return True
    else:
        return False

# Declare leaf node
def make_leaf(data):
    label_column = data[:, -1]
    return np.mean(label_column)

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

# Calculate the MSE of group of data
def calculate_MSE(data):
    label_column = data[:, -1]
    if len(label_column) == 0:
        MSE = 0

    else:
        pred = np.mean(label_column)
        MSE = np.mean((pred - label_column) ** 2)

    return MSE

# Calculate total MSE of both branches
def calculate_total_MSE(left, right):
    n = len(left) + len(right)
    p_left = len(left) / n
    p_right = len(right) / n

    total_MSE = (p_left * calculate_MSE(left)
                     + p_right * calculate_MSE(right))

    return total_MSE

# Find the splits with lowest total entropy
def determine_best_split(data, potential_splits):
    # Start with an impossible high value that calc MSE won't be higher than
    first_iter = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            left, right = split_branch(data, split_column=column_index, split_value=value)
            current_total_MSE = calculate_total_MSE(left, right)

            if first_iter or current_total_MSE <= best_total_MSE:
                first_iter = False

                best_total_MSE = current_total_MSE
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
my_tree = build_decision_tree(df_none_nan, max_depth=max_depth)
print("\nMy regression tree:")
pprint(my_tree)


### Classification ###

# Import and sort test set
df_censored_test = pd.read_csv('censored_data.csv', sep=' ', header=None)
df_nan_rows = df_censored_test
df_censored_test = df_censored_test.dropna()
df_nan_rows = df_nan_rows.drop(df_censored_test.index)
df_nan_rows = df_nan_rows[cols]

df_uncensored = pd.read_csv('uncensored_data.csv', sep=' ', header=None)
df_uncensored = df_uncensored.drop(df_censored_test.index)
df_truth = df_uncensored.iloc[:1]





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


def calculate_MSE_on_test(df_test, df_truth, tree):
    pred = df_test.apply(classify_test_set, axis=1, args=(tree,))
    pred = pred.to_numpy()
    label_column = df_truth[1].to_numpy()
    MSE = np.mean((pred - label_column) ** 2)

    return MSE


# Check MSE of regression tree
MSE_test = calculate_MSE_on_test(df_nan_rows, df_truth, my_tree)
print("\nMSE for missing values using a regression tree with max depth = 3: %.3f" % MSE_test)


