import os
import math
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV, Ridge, BayesianRidge

"""
    Process all .csv files to create a dataframe that will be encoded and standardized. 

    @param csv_filepath: filepath of where all the .csv files are
    @param drop_columns: columns that will be dropped from the dataframe before further processing
    @return dataframe of scaled input data 
    @return dataframe of scaled output data 
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
    
    # Scale the input data using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(concat_df), columns = concat_df.columns)

    # Prepare for a tensor structure
    input_ = df_scaled.iloc[:, :-1] # Select every column except last column of dataframe
    output_ = df_scaled.iloc[:, -1:] # Select only last column of dataframe
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
def lasso_regression(X_train, y_train, X_test):
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
def ridge_regression(X_train, y_train, X_test):
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
def lasso_regression(X_train, y_train, X_test):
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

def plot_gt_predictions(random_gt, random_pred, title, savefig_title):
    fig = plt.figure(facecolor = 'w', figsize = (18, 10))
    plt.style.use('ggplot')
    plt.scatter(random_gt, random_pred, c = 'crimson')
    plt.yscale('log')
    plt.xscale('log')
    p1 = max(max(random_pred), max(random_gt))
    p2 = min(min(random_pred), min(random_gt))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title(title)
    plt.xlabel('True Values', fontsize = 15)
    plt.ylabel('Predictions', fontsize = 15)
    plt.axis('equal')
    plt.savefig(savefig_title)
    plt.close()

"""
    Apply principal components analysis on data passed in.

    @param data: features passed in
    @param n: number of principal components
    @return data transformed by PCA
    @return PCA to apply
"""
def create_pca(data, n):
    pca = PCA(n)
    X_transformed = pca.fit_transform(data)
    return X_transformed, pca

"""
    Tiers to place average packet loss values in.

    @param average_packet_loss: target data of BNNC dataset
    @return which tier the data point belongs to
"""
def bracket_selection(average_packet_loss):
  if average_packet_loss <= 0.1:
    return "Tier 1"
  elif average_packet_loss <= 0.2:
    return "Tier 2"
  elif average_packet_loss <= 0.4:
    return "Tier 3"
  elif average_packet_loss <= 0.6:
    return "Tier 4"
  else:
    return "Tier 5"

# Driver code
if __name__ == "__main__":
    TRAINING_PATH = 'training\\'
    TEST_PATH = 'test\\'
    DROP_COLUMNS = ['Unnamed: 0', 'Time Distribution', 'Size Distribution', 'Link Exists', 'Avg Utilization', 'Avg Packet Length', 
                'Avg Utilization First', 'Avg Packet Loss Rate', 'Avg Port Occupancy', 'Max Queue Occupancy', 'Avg Packet Length First']
    DROP_COLUMNS_FE = ['Unnamed: 0', 'Average Per-Packet Delay', 'Percentile 10', 'Percentile 20', 'Percentile 50', 
        'Percentile 80', 'Jitter', 'Exponential Max Factor', 'Average Packet Size', 'Packet Size 1', 'Packet Size 2', 
        'Time Distribution', 'Size Distribution', 'Link Exists', 'Avg Utilization', 'Avg Packet Length', 
        'Avg Utilization First', 'Avg Packet Loss Rate', 'Avg Port Occupancy', 'Max Queue Occupancy', 'Avg Packet Length First']
    X_train_fe, y_train = process_dataframe(TRAINING_PATH, DROP_COLUMNS_FE)
    X_test_fe, y_test = process_dataframe(TEST_PATH, DROP_COLUMNS_FE)

    # X_train = X_train[['Global Packet',
    #                 'Global Loss',
    #                 'Global Delay',
    #                 'Average Bandwidth',
    #                 'Packets Transmitted',
    #                 'Packets Dropped',
    #                 'Average Per-Packet Delay',
    #                 'Neperian Logarithm',
    #                 'Percentile 10',
    #                 'Percentile 20',
    #                 'Percentile 50',
    #                 'Percentile 80',
    #                 'Percentile 90',
    #                 'Jitter',
    #                 'Max Avg Lambda',
    #                 'Equivalent Lambda',
    #                 'Average Packet Lambda',
    #                 'Exponential Max Factor',
    #                 'Average Packet Size',
    #                 'Packet Size 1',
    #                 'Packet Size 2']]

    X_train_fe = X_train_fe[['Max Avg Lambda',
                        'Equivalent Lambda',
                        'Average Packet Lambda',
                        'Packets Transmitted',
                        'Average Bandwidth',
                        'Global Packet',
                        'Global Loss',
                        'Global Delay',
                        'Neperian Logarithm',
                        'Percentile 90',
                        'Packets Dropped']]

    y_train = y_train['Avg Packet Loss']

    # X_test = X_test[['Global Packet',
    #                 'Global Loss',
    #                 'Global Delay',
    #                 'Average Bandwidth',
    #                 'Packets Transmitted',
    #                 'Packets Dropped',
    #                 'Average Per-Packet Delay',
    #                 'Neperian Logarithm',
    #                 'Percentile 10',
    #                 'Percentile 20',
    #                 'Percentile 50',
    #                 'Percentile 80',
    #                 'Percentile 90',
    #                 'Jitter',
    #                 'Max Avg Lambda',
    #                 'Equivalent Lambda',
    #                 'Average Packet Lambda',
    #                 'Exponential Max Factor',
    #                 'Average Packet Size',
    #                 'Packet Size 1',
    #                 'Packet Size 2']]

    X_test_fe = X_test_fe[['Max Avg Lambda',
                    'Equivalent Lambda',
                    'Average Packet Lambda',
                    'Packets Transmitted',
                    'Average Bandwidth',
                    'Global Packet',
                    'Global Loss',
                    'Global Delay',
                    'Neperian Logarithm',
                    'Percentile 90',
                    'Packets Dropped']]

    y_test = y_test['Avg Packet Loss']
    
    y_tiers = y_train.apply(bracket_selection)

    # Perform Principal Components Analysis for feature selection
    # Decomposing the train set:
    #pca_train_results, pca_train = create_pca(X_train, 18)

    #Decomposing the test set:
    #pca_test_results, pca_test = create_pca(X_test, 18)

    # Plotting the first three PCA components and if it helps us distinguish the avg packet loss
    #first_comps = pca_train_results[:,0] #Taking the first PCA component 
    #second_comps = pca_train_results[:,1]

    # plt.figure(figsize=(16,10))
    # sns.scatterplot(
    #     x=first_comps, 
    #     y=second_comps,
    #     hue=y_tiers,
    #     palette=sns.color_palette("hls", 5),
    #     legend="full",
    #     alpha=0.3
    # )

    # plt.title("Average Packet Loss Explained by First Two PCA Elements")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.savefig('first_two_pca.png') # Training data
    # plt.close()

    # #Creating a table with the explained variance ratio
    # names_pcas = [f"PCA Component {i}" for i in range(1, 19, 1)]
    # scree = pd.DataFrame(list(zip(names_pcas, pca_train.explained_variance_ratio_)), 
    #                 columns = ["Component", "Explained Variance Ratio"])
    # print(scree)

    # # Scree plot
    # indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    # plt.scatter(indices, scree["Explained Variance Ratio"])
    # plt.xlabel('PCA Component', fontsize = 15)
    # plt.ylabel('Explained Variance Ratio', fontsize = 15)
    # plt.title('Scree Plot')
    # plt.savefig('scree_plot.png')
    # plt.close()

    # # Sorting the values of the first principal component by how large each one is
    # df = pd.DataFrame({'PCA': pca_train.components_[0], 'Variable Names': list(X_train.columns)})
    # df = df.sort_values('PCA', ascending = False)

    # # Sorting the absolute values of the first principal component by magnitude
    # df2 = pd.DataFrame(df)
    # df2['PCA'] = df2['PCA'].apply(np.absolute)
    # df2 = df2.sort_values('PCA', ascending = False)
    # print(df2['Variable Names'][0:11])
    # print(df.head())

    train_rmse = []
    test_rmse = []
 
    # Multiple linear regression and RMSE calculation    
    linear_regression, y_pred_reg = multiple_linear_regression(X_train_fe, y_train, X_test_fe)
    linear_regression_training_rmse = calculate_rmse(y_train, linear_regression.predict(X_train_fe))
    print(f'Multiple linear regression training RMSE: {linear_regression_training_rmse}')
    linear_regression_test_rmse = calculate_rmse(y_test, y_pred_reg)
    print(f'Multiple linear regression test RMSE: {linear_regression_test_rmse}')
    train_rmse.append(linear_regression_training_rmse)
    test_rmse.append(linear_regression_test_rmse)

    # LASSO regression, RMSE calculations, and attribute selection   
    lasso_regression, y_pred_lasso = lasso_regression(X_train_fe, y_train, X_test_fe)
    lasso_regression_training_rmse = calculate_rmse(y_train, lasso_regression.predict(X_train_fe))
    print(f'LASSO regression training RSME: {lasso_regression_training_rmse}')
    lasso_regression_test_rmse = calculate_rmse(y_test, y_pred_lasso)
    print(f'LASSO regression test RMSE: {lasso_regression_test_rmse}')
    print(f'LASSO regression model coefficient:\n {lasso_regression.coef_}')
    print(f'The shrinkage coefficient hyperparameter chosen by CV in LASSO regression: {lasso_regression.alpha_}')
    train_rmse.append(lasso_regression_training_rmse)
    test_rmse.append(lasso_regression_test_rmse)

    # Ridge regression, RMSE calcuations, and attribute selection
    ridge_regression, y_pred_ridge = ridge_regression(X_train_fe, y_train, X_test_fe)
    ridge_regression_training_rmse = calculate_rmse(y_train, ridge_regression.predict(X_train_fe))
    print(f'Ridge regression training RSME: {ridge_regression_training_rmse}')
    ridge_regression_test_rmse = calculate_rmse(y_test, y_pred_ridge)
    print(f'Ridge regression test RMSE: {ridge_regression_test_rmse}')
    print(f'Ridge regression model coefficient:\n {ridge_regression.coef_}')
    train_rmse.append(ridge_regression_training_rmse)
    test_rmse.append(ridge_regression_test_rmse)

    # Random forest regression and RMSE calculation
    rf_regression, y_pred_rf = random_forest_regression(X_train_fe, y_train, X_test_fe, 0, 5)
    rf_regression_training_rmse = calculate_rmse(y_train, rf_regression.predict(X_train_fe))
    print(f'Random forest regression training RMSE: {rf_regression_training_rmse}')
    rf_regression_test_rmse = calculate_rmse(y_test, y_pred_rf)
    print(f'Random forest regression test RMSE: {rf_regression_test_rmse}')
    train_rmse.append(rf_regression_training_rmse)
    test_rmse.append(rf_regression_test_rmse)

    # Bayesian ridge regression and RMSE calculation    
    bayesian_ridge_reg, y_pred_br = bayesian_ridge_regression(X_train_fe, y_train, X_test_fe)
    bayesian_regression_training_rmse = calculate_rmse(y_train, bayesian_ridge_reg.predict(X_train_fe))
    print(f'Bayesian ridge regression training RMSE: {bayesian_regression_training_rmse}')
    bayesian_regression_test_rmse = calculate_rmse(y_test, y_pred_br)
    print(f'Bayesian ridge regression test RMSE: {bayesian_regression_test_rmse}')
    train_rmse.append(bayesian_regression_training_rmse)
    test_rmse.append(bayesian_regression_test_rmse)

    regression_dictionary_training = {'Multiple linear regression': linear_regression.predict(X_train_fe),
                                    'LASSO regression': lasso_regression.predict(X_train_fe),
                                    'Ridge regression': ridge_regression.predict(X_train_fe),
                                    'Random forest regression': rf_regression.predict(X_train_fe),
                                    'Bayesian ridge regression': bayesian_ridge_reg.predict(X_train_fe)}

    regression_dictionary_test = {'Multiple linear regression': y_pred_reg,
                                'LASSO regression': y_pred_lasso,
                                'Ridge regression': y_pred_ridge,
                                'Random forest regression': y_pred_rf,
                                'Bayesian ridge regression': y_pred_br}

    # Take a random sample of points
    idx = np.random.randint(0, y_train.shape[0], 500) 
    random_sample_mlr = regression_dictionary_training['Multiple linear regression'][idx]
    random_sample_lasso = regression_dictionary_training['LASSO regression'][idx]
    random_sample_ridge = regression_dictionary_training['Ridge regression'][idx]
    random_sample_rf = regression_dictionary_training['Random forest regression'][idx]
    random_sample_br = regression_dictionary_training['Bayesian ridge regression'][idx]
    random_y_train = y_train[idx]

    # Plot actual values against predictions, and correlation   
    plot_gt_predictions(random_y_train, 
                    random_sample_mlr, 
                    'Multiple Linear Regression Training Set Comparison between Ground Truth and Predicted Values',
                    "images\\mlr_train_fe.png")
    
    plot_gt_predictions(random_y_train,
                    random_sample_lasso,
                    'LASSO Regression Training Set Comparison between Ground Truth and Predicted Values',
                    "images\\lasso_train_fe.png") 

    plot_gt_predictions(random_y_train,
                    random_sample_ridge,
                    'Ridge Regression Training Set Comparison between Ground Truth and Predicted Values',
                    "images\\ridge_train_fe.png")

    plot_gt_predictions(random_y_train,
                    random_sample_rf,
                    'Random Forest Regression Training Set Comparison between Ground Truth and Predicted Values',
                    "images\\rf_train_fe.png")
      
    plot_gt_predictions(random_y_train,
                    random_sample_br,
                    'Bayesian Ridge Regression Training Set Comparison between Ground Truth and Predicted Values',
                    "images\\br_train_fe.png")

    idx = np.random.randint(0, y_test.shape[0], 500) 
    random_sample_mlr = regression_dictionary_test['Multiple linear regression'][idx]
    random_sample_lasso = regression_dictionary_test['LASSO regression'][idx]
    random_sample_ridge = regression_dictionary_test['Ridge regression'][idx]
    random_sample_rf = regression_dictionary_test['Random forest regression'][idx]
    random_sample_br = regression_dictionary_test['Bayesian ridge regression'][idx]
    random_y_test = y_test[idx]
     
    plot_gt_predictions(random_y_test, 
                    random_sample_mlr, 
                    'Multiple Linear Regression Test Set Comparison between Ground Truth and Predicted Values',
                    "images\\mlr_test_fe.png")
    
    plot_gt_predictions(random_y_test,
                    random_sample_lasso,
                    'LASSO Regression Test Set Comparison between Ground Truth and Predicted Values',
                    "images\\lasso_test_fe.png")

    plot_gt_predictions(random_y_test,
                    random_sample_ridge,
                    'Ridge Regression Test Set Comparison between Ground Truth and Predicted Values',
                    "images\\ridge_test_fe.png")

    plot_gt_predictions(random_y_test,
                    random_sample_rf,
                    'Random Forest Regression Test Set Comparison between Ground Truth and Predicted Values',
                    "images\\rf_test_fe.png")
        
    plot_gt_predictions(random_y_test,
                    random_sample_br,
                    'Bayesian Ridge Regression Test Set Comparison between Ground Truth and Predicted Values',
                    "images\\br_test_fe.png")

    # Create visualization of RMSE results    
    labels = ['Linear', 'LASSO', 'Ridge', 'Random Forest', 'Bayesian Ridge']
    x = np.arange(len(labels)) # Label locations
    width = 0.5 # Width of bars
    fig, ax = plt.subplots(figsize = (15, 5))
    rect_1 = ax.bar(x + 0.00, train_rmse, width, color = 'r', label = "Training")
    rect_2 = ax.bar(x + 0.25, test_rmse, width, color = 'b', label = "Test")
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error for Regression of BNNC Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('rmse_comparison_fe.png')
    plt.close()