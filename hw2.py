import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)


df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv')

#subset of columns 
df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year','fuel_efficiency_mpg']]

# columns or column with missing values
df.isna().sum()>0

#median of horsepower column
df.horsepower.median()

# Function for splitting and shuffling data
def shuffle_data(df, seed):
    """
    This function splits the data into train - validation - test sets with
    respectively 60% - 20% - 20% of the original dataframe, shuffles the obtained sets,
    and returns them with new index.
    ---
    df: dataframe
    seed: numpy seed for reproducibility
    """
    # Data set size
    n = len(df)
    # Validation and Test Size
    n_val = n_test = int(n * 0.2)
    # Train set size
    n_train = n - (n_val * 2)
    
    # Create an index
    idx = np.arange(n)
    # Shuffle the index
    np.random.seed(seed) # for reproducibility
    np.random.shuffle(idx)

    # Train - Val - Test splitting
    df_train = df.iloc[idx[ : n_train]]
    df_val = df.iloc[idx[n_train : n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val : ]]

    # Reset indexes
    df_train.reset_index(drop = True, inplace = True)
    df_val.reset_index(drop = True, inplace = True)
    df_test.reset_index(drop = True, inplace = True)

    # return shuffled data
    return df_train, df_val, df_test

# Train - Validation - test splitting with a seed of 42
df_train, df_val, df_test = shuffle_data(df = df, seed = 42)


# Data Preparation Function
def prepare_X(df, fill_value = 0):
    """
    This function prepares a feature matrix `X`.
    ---
    df: dataframe,
    fill_value: value used for filling missing values if any.
        default: `0`
        Otherwise, please preferably enter a dictionnary or a single value.
    """
    # Remove target feature
    df_num = df.drop(columns = "fuel_efficiency_mpg")

    # Filling missing values
    df_num = df_num.fillna(fill_value)

    # Extract an array
    X = np.array(df_num)

    # return feature matrix
    return X

# Linear regression model traininng
def train_linear_regression(X, y, r = 0):
    """
    Function to train a linear regression model.
    ---
    X: Feature matrix
    y: target vector
    r: regularization parameter if any,
        defalut value: 0 (no regularization)
    """
    # X_0 vector
    ones = np.ones(X.shape[0])   
    # Feature matrix
    X = np.column_stack([ones, X])
    
    # Gram matrix
    XTX = X.T.dot(X)
    # Regularization
    XTX = XTX + r * np.eye(XTX.shape[0])
    # Gram matrix inverse
    XTX_inv = np.linalg.inv(XTX)
    
    # Weights
    w_full = XTX_inv.dot(X.T).dot(y)

    # return slope and coefficients
    return w_full[0], w_full[1 :]

# Root Mean Square Root Error Function
def rmse(y, y_pred):
    """
    Root Mean Square Error between real values and model predictions.
    ---
    y: real values
    y_pred: model predictions
    """
    # Square errors
    se = (y - y_pred) ** 2

    # Mean square error
    mse = se.mean()

    # return root mean square error
    return np.sqrt(mse)

# Funtion to evaluate the model
def evaluate_model(train, val, fill_value = 0, r = 0):
    """
    This function evaluates a linear regression model by returning a rounded rmse.
    ---
    train: training dataframe 
    val: validation dataframe
    fill_value: value used for filling missing values if any,
        default: `0`,
        Otherwise, please preferably enter a dictionnary or a single value
    r: regularization parameter for linear regression
    """
    # Extract targets
    y_train = train.fuel_efficiency_mpg.values
    y_val = val.fuel_efficiency_mpg.values
    
    # Prepare training data
    X_train = prepare_X(train, fill_value)
    
    # Linear Regression
    w0, w = train_linear_regression(X_train, y_train, r)
    
    # Prepare validation data
    X_val = prepare_X(val, fill_value)
    
    # Make predictions on validation data
    y_pred = w0 + X_val.dot(w)
    
    # Rounded RMSE on validation data
    score = rmse(y_val, y_pred)

    # return score
    return round(score, 2)

# Model RMSE score when filling missing values with 0
evaluate_model(train = df_train, val = df_val, fill_value = 0)

# Model RMSE score when filling missing values with columns' means
evaluate_model(train = df_train, val = df_val, fill_value = df_train.mean().to_dict())

# Bunch of values for the hyperparameter
params = [0, 0.01, 0.1, 1, 5, 10, 100]

# Initialize results
results = []

# Model Fine Tuning with parameters
for param in params:
    # store model evaluation score into results
    results += [evaluate_model(train = df_train, val = df_val, fill_value = 0, r = param)]


pd.Series(data = results, index = params, name = "parameters_results").sort_values()

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
# Initialize results
results_seeds = []
# Model evaluation with different seeds
for seed in seeds:
    # Train - Val - Test splitting
    df_train, df_val, df_test = shuffle_data(df = df, seed = seed)
    # store model evaluation score into results
    results_seeds += [evaluate_model(train = df_train, val = df_val, fill_value = 0, r = 0.1)]

# std of results rounded to 3 decimal places
round(np.std(results_seeds), 3)

# Model evaluation with test set
# Train - Validation - test with a seed of `9`
df_train, df_val, df_test = shuffle_data(df = df, seed = 9)

# Combine Training and Validation sets
df_full_train = pd.concat([df_train, df_val])

# model evaluation score on the test dataset
evaluate_model(train = df_full_train, val = df_test, fill_value = 0, r = 0.001)

