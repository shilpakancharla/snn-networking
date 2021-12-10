import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import utils, spikegen
from snntorch import spikeplot as splt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

class SpikingNeuralNetwork(nn.Module):
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

def print_batch_accuracy(data, targets, train = False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim = 0).max(1)
    print(type(idx))
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train = True)
    print_batch_accuracy(test_data, test_targets, train = False)
    print("\n")

"""
    Process all .csv files to create a dataframe that will be encoded and standardized. Convert this structure to a tensor
    and shape it into the required format for the SNN.

    @param csv_filepath: filepath of where all the .csv files are
    @param drop_columns: columns that will be dropped from the dataframe before further processing
    @return numpy array of input data in appropriate SNN dimensions
    @return numpy array of output data in appropriate SNN dimensions
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

    # Scale the input data using StandardScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(concat_df), columns = concat_df.columns)
    
    # Prepare for a tensor structure
    input_ = df_scaled.iloc[:, :-1].values # Select every column except last three columns of dataframe
    output_ = df_scaled.iloc[:, -1:].values # Select only last three columns of dataframe
    return input_, output_

def create_output_histogram(numerical_data_array, title):
    bins = np.linspace(0, 1, 100, endpoint = True)
    # Average packet loss is [0, 1]
    _ = plt.hist(numerical_data_array, bins = bins, weights = numerical_data_array)
    plt.title(title)
    plt.xlabel("Average Packet Loss")
    plt.ylabel("Number of Values")
    plt.show()

def rate_coding(tensor, batch_size, number_steps):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [time x batch_size x feature_dimensions]
    spike_data = spikegen.rate(data_it, num_steps = number_steps)
    return spike_data

def latency_coding(tensor, batch_size, number_steps, tau, threshold):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [time x batch_size x feature_dimensions]
    spike_data = spikegen.latency(data_it, num_steps = number_steps)
    return spike_data

def delta_modulation(tensor, batch_size, threshold):
    # Create DataLoader object
    data_loader = DataLoader(tensor, batch_size = batch_size, shuffle = True)
    data = iter(data_loader)
    data_it = next(data)
    
    # Create data of dimensions [batch_size x feature_dimensions]
    spike_data = spikegen.delta(data_it, threshold = threshold, off_spike = True)
    return spike_data

def training_one_iteration(train_loader, dtype, device, optimizer):
    # One iteration of training
    data, targets = next(iter(train_loader))
    print(data.type())
    data = data.to(device)
    targets = targets.to(device)
    num_steps = 25
    spk_rec, mem_rec = net(data.view(batch_size, -1))
    print(mem_rec.size())

    # Initailize the total loss value
    loss_val = torch.zeros((1), dtype = dtype, device = device)

    # Sum loss at every step
    for step in range(num_steps):
        loss_val += loss(mem_rec[step], targets)

    print(f"Training loss: {loss_val.item():.3f}")

    print_batch_accuracy(data, targets, train = True)
    
    
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
        loss_val += loss(mem_rec[step], targets)

    print(f"Training loss: {loss_val.item():.3f}")
    print_batch_accuracy(data, targets, train = True)

def training_loop(net, train_loader, test_loader, dtype, device, optimizer, loss_val):
    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0

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
            spk_rec, mem_rec = net(data.view(batch_size, -1))

            # Initialize the loss and sum over time
            loss_val = torch.zeros((1), dtype = dtype, device = device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation and weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype = dtype, device = device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], target_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss and accuracy
                if counter % 50 == 0:
                    train_printer()
                counter = counter + 1
                iter_counter = iter_counter + 1
    
    return loss_hist, test_loss_hist

# Driver code
if __name__ == "__main__":
    TRAINING_PATH = 'training\\'
    TEST_PATH = 'test\\'
    DROP_COLUMNS = ['Unnamed: 0', 'Time Distribution', 'Size Distribution', 'Link Exists', 'Avg Utilization', 'Avg Packet Length', 
                'Avg Utilization First', 'Avg Packet Loss Rate', 'Avg Port Occupancy', 'Max Queue Occupancy', 'Avg Packet Length First']
    #input_train, output_train = process_dataframe(TRAINING_PATH, DROP_COLUMNS)
    #input_test, output_test = process_dataframe(TEST_PATH, DROP_COLUMNS)
    #print(input_train.shape)
    #print(output_train.shape)
    #print(input_test.shape)
    #print(output_test.shape)
    
    # Save these numpy arrays to load in again
    #np.save("input_train.npy", input_train)
    #np.save("output_train.npy", output_train)
    #np.save("input_test.npy", input_test)
    #np.save("output_test.npy", output_test)

    # Load .npy files once you save them
    INPUT_TRAIN = 'input_train.npy'
    OUTPUT_TRAIN = 'output_train.npy'
    INPUT_TEST = 'input_test.npy'
    OUTPUT_TEST = 'output_test.npy'
    features_train_tensor = np.load(INPUT_TRAIN)
    target_train_tensor = np.load(OUTPUT_TRAIN)
    print(len(target_train_tensor))
    # Bin the output train data
    #create_output_histogram(target_train_tensor, "Histogram for range of values for training output")
    features_test_tensor = np.load(INPUT_TEST)
    target_test_tensor = np.load(OUTPUT_TEST)
    # Bin the output test data
    create_output_histogram(target_test_tensor, "Histogram for range of values for test output")

    batch_size = 128

    # Passing numpy array to to DataLoader
    train = TensorDataset(torch.from_numpy(features_train_tensor).float(), torch.from_numpy(target_train_tensor).float())
    test = TensorDataset(torch.from_numpy(features_test_tensor).float(), torch.from_numpy(target_test_tensor).float())
    train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test, batch_size = batch_size, shuffle = True)

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = SpikingNeuralNetwork(21, 1000, 1, 0.95).to(device) # Load network onto CUDA if available

    loss_function = nn.MSELoss() # Regression mean squared loss
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-4, betas = (0.9, 0.999))

    training_one_iteration(train_loader, dtype, device, optimizer)