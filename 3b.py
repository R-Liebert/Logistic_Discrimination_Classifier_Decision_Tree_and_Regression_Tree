import numpy as np
import pandas as pd

# Import file
df_censored = pd.read_csv('censored_data.csv', sep=' ', header=None)
df_uncensored = pd.read_csv('uncensored_data.csv', sep=' ', header=None)

# Check which column that has missing values (nan)
nan_values = df_censored.isna()
columns_with_nan = nan_values.any()

list_columns_with_nan = df_censored.columns[columns_with_nan].tolist()
if len(list_columns_with_nan) == 1:
      list_columns_with_nan = list_columns_with_nan[0]
      n = df_censored.isnull().sum(axis = 0)
      n = n[list_columns_with_nan]

print("There is {} NaN-values in column {}.".format(n, list_columns_with_nan))

# Find mean, median and max value of column with nan values
nan_column_mean = df_censored[list_columns_with_nan].mean()

nan_column_median = df_censored[list_columns_with_nan].median()

nan_column_max = df_censored[list_columns_with_nan].max()

# Show the different values of the column with nan
print("\nValues for column {}:\n Mean value: {:.3f}\n Median value: {:.3f}\n Max value: {:.3f}\n"
      .format(list_columns_with_nan, nan_column_mean, nan_column_median, nan_column_max))

# Create new dataframes and fill missing values with mean, median and max values
df_mean_fill = df_censored
mean_fill_series = df_mean_fill[list_columns_with_nan].replace(np.nan, nan_column_mean)

df_median_fill = df_censored
median_fill_series = df_median_fill[list_columns_with_nan].replace(np.nan, nan_column_median)

df_max_fill = df_censored
max_fill_series = df_max_fill[list_columns_with_nan].replace(np.nan, nan_column_max)


# getting the true values
df_y = df_uncensored
non_nan_df = df_censored.dropna()
df_y = df_y.drop(non_nan_df.index)

y = df_y[1]

# Computing the Mean Square Error of the imputed values
y_mean = np.ones(len(y)) * nan_column_mean
y_median = np.ones(len(y)) * nan_column_median
y_max = np.ones(len(y)) * nan_column_max

MSE_mean = np.sum((y_mean - y)**2) / n
MSE_median = np.sum((y_median - y)**2) / n
MSE_max = np.sum((y_max - y)**2) / n

print("MSE for the missing values when replaced with the mean = {0:.2f}, median = {1:.2f} and max = {2:.2f}. "
      .format(MSE_mean, MSE_median, MSE_max))

#
correlation_matrix = df_censored.corr()

print("\nCorrelation matrix for censored data:\n", correlation_matrix)

# Observations and training beta
df_train = df_censored.dropna()
X = np.c_[np.ones((df_train.shape[0],1)), df_train[0]]
y = df_train[1]
beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
print("\nEstimated coefficients:\nb_0 = {:.3f} \nb_1 = {:.3f}".format(beta[0],
beta[1]))

# creating array of the rows with missing values
df_x = df_censored
df_x = df_x.drop(df_train.index)

# creating a vector of the values in column 0 where there is missing values in column 1
x = df_x[0]

# estimating missing parameters
y_pred = beta[0] + x*beta[1]
y_pred = y_pred.to_numpy()

# getting the true values
df_y = df_uncensored
df_y = df_y.drop(df_train.index)

y = df_y[0].to_numpy()

# Calculate MSE
MSE_reg = np.sum((y_pred - y)**2) / y.shape[0]

print("\nMSE for the missing values when replaced using regression on the first column = %.2f." % MSE_reg)