import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import utils, spikegen
from snntorch import spikeplot as splt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

"""
    Author: Shilpa Kancharla
    Last Modified: December 15, 2021
"""

class SpikingLeakyNeuralNetwork(nn.Module):
    """
        Parameters in SpikingLeakyNeuralNetwork class:
        
        1. number_inputs: Number of inputs to the SNN.
        2. number_hidden: Number of hidden layers.
        3. number_outputs: Number of output classes.
        4. beta: Decay rate. 
    """
    def __init__(self, number_inputs, number_hidden, number_outputs, beta):
        super().__init__()
        self.number_inputs = number_inputs
        self.number_hidden = number_hidden
        self.number_outputs = number_outputs
        self.beta = beta

        # Initialize layers
        self.fc1 = nn.Linear(self.number_inputs, self.number_hidden) # Applies linear transformation to all input points
        self.lif1 = snn.Leaky(beta = self.beta) # Integrates weighted input over time, emitting a spike if threshold condition is met
        self.fc2 = nn.Linear(self.number_hidden, self.number_outputs) # Applies linear transformation to output spikes of lif1
        self.lif2 = snn.Leaky(beta = self.beta) # Another spiking neuron, integrating the weighted spikes over time

    """
        Forward propagation of SNN. The code below function will only be called once the input argument x 
        is explicitly passed into net.

        @param x: input passed into the network
        @return layer of output after applying final spiking neuron
    """
    def forward(self, x):
        num_steps = 25

        # Initialize hidden states at t = 0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim = 0), torch.stack(mem2_rec, dim = 0)

"""
    Calculate the accuracy after each iteration for the train and test sets.

    @param data: feature values
    @param targets: target values
    @param train: Boolean of if we are in train mode or not
    @return accuracy value for the iteration
"""
def print_batch_accuracy(data, targets, train = False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim = 0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")
    return acc * 100

"""
    Print the results of training. 

    @param epoch: which epoch is occuring right now
    @param iter_counter: counts number of iterations
    @param counter: indexes what content to print in loss history
    @param loss_history: array of loss values
    @param data: feature values of training set
    @param targets: target values of training set
    @param test_data: feature values of test set
    @param test_targets: target values of test set
"""
def train_printer(epoch, iter_counter, counter, loss_history, data, targets, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_history[counter]:.2f}")
    print(f"Test Set Loss: {loss_history[counter]:.2f}")
    acc = print_batch_accuracy(data, targets, train = True)
    test_acc = print_batch_accuracy(test_data, test_targets, train = False)
    print("\n")
    return acc, test_acc

"""
    Process all .csv files to create a dataframe that will be encoded and standardized. Convert this structure to a tensor
    and shape it into the required format for the SNN.

    @param csv_filepath: filepath of where all the .csv files are
    @param drop_columns: columns that will be dropped from the dataframe before further processing
    @return numpy array of scaled input data in appropriate SNN dimensions
    @return numpy array of scaled output data in appropriate SNN dimensions
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
    input_ = df_scaled.iloc[:, :-1].values # Select every column except last column of dataframe
    output_ = df_scaled.iloc[:, -1:].values # Select only last column of dataframe
    return input_, output_

"""
    Create bins of categorical data for average packet loss.

    @param numerical_data_array: numpy array of output day
    @return binned output data into categories
"""
def create_bins(numerical_data_array):
    bins = np.linspace(0, 1, 100, endpoint = True) # Average packet loss is [0, 1]
    idxs = np.digitize(numerical_data_array, bins)
    return idxs

"""
    Creates histograms based on bins of output data of average packet loss.

    @param numerical_data_array: numpy array of output day
    @param title: title of histogram
"""
def create_output_histogram(numerical_data_array, title):
    bins = np.linspace(0, 1, 100, endpoint = True) # Average packet loss is [0, 1]
    _ = plt.hist(numerical_data_array, bins = bins, weights = numerical_data_array)
    plt.title(title)
    plt.xlabel("Average Packet Loss")
    plt.ylabel("Number of Values")
    plt.show()

"""
    Rate coding for temporal input.

    @param tensor: tensor input and target data to be coded
    @param batch_size: size of batch to be fed into neural network
    @param number_steps: temporal dynamics
    @return rate spike coded data
"""
def rate_coding(tensor, batch_size, number_steps):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [time x batch_size x feature_dimensions]
    spike_data = spikegen.rate(data_it, num_steps = number_steps)
    return spike_data

"""
    Latency coding for temporal input.

    @param tensor: tensor input and target data to be coded
    @param batch_size: size of batch to be fed into neural network
    @param number_steps: temporal dynamics
    @tau: he RC time constant of the circuit; higher tau will induce slower firing
    @threshold: the membrane potential firing threshold
    @return latency spike coded data
"""
def latency_coding(tensor, batch_size, number_steps, tau, threshold):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [time x batch_size x feature_dimensions]
    spike_data = spikegen.latency(data_it, num_steps = number_steps)
    return spike_data

"""
    Rate coding for temporal input.

    @param tensor: tensor input and target data to be coded
    @param batch_size: size of batch to be fed into neural network
    @param number_steps: temporal dynamics
    @return delta modulated coded data
"""
def delta_modulation(tensor, batch_size, threshold):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [batch_size x feature_dimensions]
    spike_data = spikegen.delta(data_it, threshold = threshold, off_spike = True)
    return spike_data

"""
    Testing out one iteration of training to ensure network can run.

    @param train_loader: DataLoader object with training data
    @param dtype: data type
    @param device: device to load network on
    @param optimizer: Adam optimizer
"""
def training_one_iteration(train_loader, dtype, device, optimizer):
    # One iteration of training
    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)
    num_steps = 25
    spk_rec, mem_rec = net(data.view(batch_size, -1))
    print(mem_rec.size())

    # Initailize the total loss value
    loss_val = torch.zeros((1), dtype = dtype, device = device)

    # Sum loss at every step
    for step in range(num_steps):
        loss_val += loss_function(mem_rec[step], targets.long().flatten().to(device))

    print(f"Training loss: {loss_val.item():.3f}")

    acc = print_batch_accuracy(data, targets, train = True)
    
    
    optimizer.zero_grad() # Clear previously stored gradients
    loss_val.backward() # Calculate the gradient
    optimizer.step() # Weight update

    # Rerun the loss calculation and accuracy after a single iteration
    # calculate new network outputs using the same data
    spk_rec, mem_rec = net(data.view(batch_size, -1))

    # initialize the total loss value
    loss_val = torch.zeros((1), dtype=dtype, device=device)

    # sum loss at every step
    for step in range(num_steps):
        loss_val += loss_function(mem_rec[step], targets.long().flatten().to(device))

    print(f"Training loss: {loss_val.item():.3f}")
    acc = print_batch_accuracy(data, targets, train = True)

"""
    Testing out one iteration of training to ensure network can run.

    @param net: spiking neural network object
    @param train_loader: DataLoader object with training data
    @param test_loader: DataLoader object with test data
    @param dtype: data type
    @param device: device to load network on
    @param optimizer: Adam optimizer
    @return loss history of train data
    @return loss history of test data
    @return accuracy history of train data (dictionary)
    @return accuracy history of test data (dictionary)
"""
def training_loop(net, train_loader, test_loader, dtype, device, optimizer):
    num_epochs = 1
    loss_history = []
    test_loss_history = []
    acc_history = dict()
    test_acc_history = dict()
    counter = 0
    count_test_loss = 0

    # Temporal dynamics
    num_steps = 25

    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            net.train()
            try:
                spk_rec, mem_rec = net(data.view(batch_size, 11))
            except RuntimeError:
                print("Hit RuntimeError.")
                return loss_history, test_loss_history, acc_history, test_acc_history # Return values to this point

            # Initialize the loss and sum over time
            loss_val = torch.zeros((1), dtype = dtype, device = device)
            for step in range(num_steps):
                loss_val += loss_function(mem_rec[step], targets.float().flatten().to(device).unsqueeze(1))

            # Gradient calculation and weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_history.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                try: 
                    test_spk, test_mem = net(test_data.view(batch_size, 11))
                except RuntimeError:
                    print("Hit RuntimeError.")
                    return loss_history, test_loss_history, acc_history, test_acc_history

                # Test set loss
                test_loss = torch.zeros((1), dtype = dtype, device = device)
                for step in range(num_steps):
                    test_loss += loss_function(test_mem[step], test_targets.float().flatten().to(device).unsqueeze(1))
                test_loss_history.append(test_loss.item())

                # Print train/test loss and accuracy
                if counter % 50 == 0:
                    acc_value, test_acc_value = train_printer(epoch, iter_counter, counter, loss_history, 
                                            data, targets, test_data, test_targets)
                    acc_history[iter_counter] = acc_value
                    test_acc_history[iter_counter] = test_acc_value
                
                counter = counter + 1
                iter_counter = iter_counter + 1

                # Break loop if any of these loss criteria are met
                if torch.allclose(test_loss, torch.tensor([0.0009])):
                    count_test_loss = count_test_loss + 1

                if count_test_loss == 3:
                    return loss_history, test_loss_history, acc_history, test_acc_history
    
    return loss_history, test_loss_history, acc_history, test_acc_history

"""
    Plot the loss and test loss histories.

    @param loss_history: loss history of train data
    @param test_loss_history: loss history of test data
"""
def plot_loss(loss_history, test_loss_history):
    fig = plt.figure(facecolor = 'w', figsize = (20, 10))
    plt.plot(loss_history)
    plt.plot(test_loss_history)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig('fe_loss_histories.png') 

"""
    Plot the accuracy history.

    @param acc: accuracy history of train data (dictionary)
    @param test_acc: accuracy history of test data (dictionary)
"""
def plot_accuracy(acc, test_acc):
    fig = plt.figure(facecolor = 'w', figsize = (20, 10))
    acc_list = acc.items()
    acc_list = sorted(acc_list)
    x_acc, y_acc = zip(*acc_list)
    plt.plot(x_acc, y_acc, label = "Training")
    test_acc_list = test_acc.items()
    test_acc_list = sorted(test_acc_list)
    test_x_acc, test_y_acc = zip(*test_acc_list)
    plt.plot(test_x_acc, test_y_acc, label = "Test")
    plt.title("Accuracy Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy %")
    plt.savefig('fe_accuracies.png')  

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
    #input_train, output_train = process_dataframe(TRAINING_PATH, DROP_COLUMNS_FE)
    #input_test, output_test = process_dataframe(TEST_PATH, DROP_COLUMNS_FE)
    
    # Check the shape of features and targets for train and test sets
    #print(input_train.shape)
    #print(output_train.shape)
    #print(input_test.shape)
    #print(output_test.shape)
    
    # Save these numpy arrays to load in again
    #np.save("npy_files\\input_train_fe.npy", input_train)
    #np.save("npy_files\\output_train_fe.npy", output_train)
    #np.save("npy_files\\input_test_fe.npy", input_test)
    #np.save("npy_files\\output_test_fe.npy", output_test)

    # Load .npy files once you save them
    INPUT_TRAIN = 'npy_files\\input_train_fe.npy'
    OUTPUT_TRAIN = 'npy_files\\output_train_fe.npy'
    INPUT_TEST = 'npy_files\\input_test_fe.npy'
    OUTPUT_TEST = 'npy_files\\output_test_fe.npy'
    features_train_tensor = np.load(INPUT_TRAIN)
    target_train_tensor = np.load(OUTPUT_TRAIN)
    
    # Bin the output train data
    #create_output_histogram(target_train_tensor, "Histogram for range of values for training output")
    #idx_train = create_bins(target_train_tensor)
    
    features_test_tensor = np.load(INPUT_TEST)
    target_test_tensor = np.load(OUTPUT_TEST)
    
    # Bin the output test data
    #create_output_histogram(target_test_tensor, "Histogram for range of values for test output")
    #idx_test = create_bins(target_test_tensor)
    
    batch_size = 128

    # Passing numpy array to to DataLoader
    train = TensorDataset(torch.from_numpy(features_train_tensor).float(), 
                        torch.from_numpy(target_train_tensor).float())
    test = TensorDataset(torch.from_numpy(features_test_tensor).float(), 
                        torch.from_numpy(target_test_tensor).float())
    train_loader = DataLoader(dataset = train, 
                            batch_size = batch_size, 
                            shuffle = True, 
                            drop_last = True)
    test_loader = DataLoader(dataset = test, 
                            batch_size = batch_size, 
                            shuffle = True, 
                            drop_last = True)

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = SpikingLeakyNeuralNetwork(11, 1000, 1, 0.95).to(device) # Load network onto CUDA if available

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-4, betas = (0.9, 0.999))

    #training_one_iteration(train_loader, dtype, device, optimizer)
    loss_history, test_loss_history, acc, test_acc = training_loop(net, train_loader, test_loader, dtype, 
                                                                device, optimizer)

    plot_loss(loss_history, test_loss_history)
    plot_accuracy(acc, test_acc)