import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, BayesianRidge

"""
    Process all .csv files to create a dataframe that will be encoded and standardized. 

    @param csv_filepath: filepath of where all the .csv files are
    @param drop_columns: columns that will be dropped from the dataframe before further processing
    @return dataframe of input data 
    @return dataframe of output data 
"""
def process_dataframe(csv_filepath, drop_columns):
    frames = []
    count = 0
    concat_df = pd.DataFrame() # Initialize an empty dataframe
    for root, directories, file_list in os.walk(csv_filepath):
        for file in file_list:
            print("Processing " + file)
            temp_df = pd.read_csv(csv_filepath + file)
            # If the link does not exist, discard the row for now
            temp_df = temp_df[temp_df['Link Exists'] != False].reset_index(drop = True)
            # Outputs of interest: Avg Packet Loss, Avg Utilization, Max Queue Occupancy
            temp_concat_df = temp_df.drop(drop_columns, axis = 1)
            # Start concatenating the files directly
            concat_df = pd.concat([concat_df.reset_index(drop = True), temp_concat_df.reset_index(drop = True)], ignore_index = True)
            count = count + 1 # Keep track of number of files processed
            print("Processed " + file + ". Compeleted " + str(count) + " of " + str(len(file_list)) + " files.")
    
    # Prepare for a tensor structure
    input_ = concat_df.iloc[:, :-1] # Select every column except last column of dataframe
    output_ = concat_df.iloc[:, -1:] # Select only last column of dataframe
    return input_, output_

"""
    Run multiple linear regression.

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @return linear regression model
    @return predictions from model
"""
def multiple_linear_regression(X_train, y_train, X_test):
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    return linear_regression, y_pred

"""
    Run LASSO regression.

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @return LASSO regression model
    @return predictions from model
"""
def lasso_regression(X_train, y_train):
    lasso_regression = LassoCV(cv = 10, random_state = 0)
    lasso_regression.fit(X_train, y_train)
    y_pred = lasso_regression.predict(X_test)
    return lasso_regression, y_pred

"""
    Run ridge regression.

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @return ridge regression model
    @return predictions from model
"""
def ridge_regression(X_train, y_train):
    ridge_regression = Ridge(alpha = 1.0)
    ridge_regression.fit(X_train, y_train)
    y_pred = ridge_regression.predict(X_test)
    return ridge_regression, y_pred

"""
    Run LASSO regression.

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @return LASSO regression model
    @return predictions from model
"""
def lasso_regression(X_train, y_train):
    lasso_regression = LassoCV(cv = 10, random_state = 0)
    lasso_regression.fit(X_train, y_train)
    y_pred = lasso_regression.predict(X_test)
    return lasso_regression, y_pred

"""
    Run random forest regression (non-linear method).

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @param random_state: seeded state
    @param max_depth: depth of regression tree
    @return random forest regression model
    @return predictions from model
"""
def random_forest_regression(X_train, y_train, X_test, random_state, max_depth):
    rf_regression = RandomForestRegressor(max_depth = max_depth, random_state = random_state)
    rf_regression.fit(X_train, y_train)
    y_pred = rf_regression.predict(X_test)
    return rf_regression, y_pred

"""
    Run Bayesian ridge regression.

    @param X_train: training set of features
    @param y_train: ground truth for training set
    @param X_test: validation set of features
    @return Bayesian ridge regression model
    @return predictions from model
"""
def bayesian_ridge_regression(X_train, y_train, X_test):
    bayesian_ridge_reg = BayesianRidge(compute_score = True)
    bayesian_ridge_reg.fit(X_train, y_train)
    y_pred = bayesian_ridge_reg.predict(X_test)
    return bayesian_ridge_reg, y_pred

"""
    Calculate the root mean squared error of a regression model to evaluate the performance.
    @param y_true: ground truth of the input
    @param y_pred: predictions for the input
    @return RMSE value
"""
def calculate_rmse(y_true, y_pred):
    return (math.sqrt(np.sum((y_true - y_pred)**2) / len(y_true)))

# Driver code
if __name__ == "__main__":
    TRAINING_PATH = 'training\\'
    TEST_PATH = 'test\\'
    DROP_COLUMNS = ['Unnamed: 0', 'Time Distribution', 'Size Distribution', 'Link Exists', 'Avg Utilization', 'Avg Packet Length', 
                'Avg Utilization First', 'Avg Packet Loss Rate', 'Avg Port Occupancy', 'Max Queue Occupancy', 'Avg Packet Length First']
    X_train, y_train = process_dataframe(TRAINING_PATH, DROP_COLUMNS)
    X_test, y_test = process_dataframe(TEST_PATH, DROP_COLUMNS)

    train_rmse = []
    test_rmse = []
 
    # Multiple linear regression and RMSE calculation
    linear_regression, y_pred_reg = multiple_linear_regression(X_train, y_train, X_test)
    linear_regression_training_rmse = calculate_rmse(y_train, linear_regression.predict(X_train))
    print(f'Multiple linear regression training RMSE: {linear_regression_training_rmse}')
    linear_regression_test_rmse = calculate_rmse(y_test, y_pred_reg)
    print(f'Multiple linear regression test RMSE: {linear_regression_test_rmse}')
    train_rmse.append(linear_regression_training_rmse)
    test_rmse.append(linear_regression_test_rmse)

    # LASSO regression, RMSE calculations, and attribute selection
    lasso_regression, y_pred_lasso = lasso_regression(X_train, y_train, X_test)
    lasso_regression_training_rmse = calculate_rmse(y_train, lasso_regression.predict(X_train))
    print(f'LASSO regression training RSME: {lasso_regression_training_rmse}')
    lasso_regression_test_rmse = calculate_rmse(y_test, y_pred_lasso)
    print(f'LASSO regression test RMSE: {lasso_regression_test_rmse}')
    print(f'LASSO regression model coefficient:\n {lasso_regression.coef_}')
    print(f'The shrinkage coefficient hyperparameter chosen by CV in LASSO regression: {lasso_regression.alpha_}')
    train_rmse.append(lasso_regression_training_rmse)
    test_rmse.append(lasso_regression_test_rmse)

    # Ridge regression, RMSE calcuations, and attribute selection
    ridge_regression, y_pred_ridge = ridge_regression(X_train, y_train, X_test)
    ridge_regression_training_rmse = calculate_rmse(y_train, ridge_regression.predict(X_train))
    print(f'Ridge regression training RSME: {ridge_regression_training_rmse}')
    ridge_regression_test_rmse = calculate_rmse(y_test, y_pred_ridge)
    print(f'Ridge regression test RMSE: {ridge_regression_test_rmse}')
    print(f'Ridge regression model coefficient:\n {ridge_regression.coef_}'))
    train_rmse.append(ridge_regression_training_rmse)
    test_rmse.append(ridge_regression_test_rmse)

    # Random forest regression and RMSE calculation
    rf_regression, y_pred_rf = random_forest_regression(X_train, y_train, X_test)
    rf_regression_training_rmse = calculate_rmse(y_train, rf_regression.predict(X_train))
    print(f'Random forest regression training RMSE: {rf_regression_training_rmse}')
    rf_regression_test_rmse = calculate_rmse(y_test, y_pred_rf)
    print(f'Random forest regression test RMSE: {rf_regression_test_rmse}')
    train_rmse.append(rf_regression_training_rmse)
    test_rmse.append(rf_regression_test_rmse)

    # Bayesian ridge regression and RMSE calculation
    bayesian_ridge_reg, y_pred_br = bayesian_ridge_regression(X_train, y_train, X_test)
    bayesian_regression_training_rmse = calculate_rmse(y_train, bayesian_ridge_reg.predict(X_train))
    print(f'Bayesian ridge regression training RMSE: {bayesian_regression_training_rmse}')
    bayesian_regression_test_rmse = calculate_rmse(y_test, y_pred_br)
    print(f'Bayesian ridge regression test RMSE: {bayesian_regression_test_rmse}')

    regression_dictionary_training = {'Multiple linear regression': linear_regression.predict(X_train),
                                    'LASSO regression': lasso_regression.predict(X_train),
                                    'Ridge regression': ridge_regression.predict(X_train),
                                    'Random forest regression': rf_regression.predict(X_train),
                                    'Bayesian ridge regression': bayesian_ridge_reg.predict(X_train)}

    regression_dictionary_test = {'Multiple linear regression': y_pred_reg,
                                'LASSO regression': y_pred_lasso,
                                'Ridge regression': y_pred_ridge,
                                'Random forest regression': y_pred_rf,
                                'Bayesian ridge regression': y_pred_br}

    # Plot actual values against predictions
    for key in regression_dictionary_training:
        plt.style.use('ggplot')
        plt.scatter(y_train, regression_dictionary_training[key], c = 'crimson')
        plt.yscale('log')
        plt.xscale('log')
        p1 = max(max(regression_dictionary_training[key]), max(y_train))
        p2 = min(min(regression_dictionary_training[key]), min(y_train))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.title(key + ' Training Set Comparison between Ground Truth and Predicted values')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.savefig(key.replace(" ", "") + "_train.png")
        plt.close()

    for key in regression_dictionary_test:
        plt.style.use('ggplot')
        plt.scatter(y_test, regression_dictionary_test[key], c = 'crimson')
        plt.yscale('log')
        plt.xscale('log')
        p1 = max(max(regression_dictionary_test[key]), max(y_test))
        p2 = min(min(regression_dictionary_test[key]), min(y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.title(key + ' Test Set Comparison between Ground Truth and Predicted values')
        plt.xlabel('True Values', fontsize=15)
        plt.ylabel('Predictions', fontsize=15)
        plt.axis('equal')
        plt.savefig(key.replace(" ", "") + "_test.png")
        plt.close()

    # Create visualization of RMSE results
    labels = ['Linear', 'LASSO', 'Ridge', 'Random Forest', 'Bayesian Ridge']

    x = np.arange(len(labels)) # Label locations
    width = 0.3 # Width of bars

    fig, ax = plt.subplots(figsize = (15, 5))
    rect_1 = ax.bar(x + 0.00, train_rmse, width, color = 'r', label = "Training")
    rect_3 = ax.bar(x + 0.25, test_rmse, width, color = 'b', label = "Test")

    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error for Barcelona Neural Networking Center Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.savefig('rmse_comparison.png')
    plt.close()