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
    def __init__(self):
        super().__init__()

        number_inputs = 21 # 21 features after feature engineering
        number_hidden = 1000
        number_outputs = 3 # 3 targets
        # Initialize layers
        fc1 = nn.Linear(number_inputs, number_hidden) # Applies linear transformation to all input points
        lif1 = snn.Leaky(beta = beta) # Integrates weighted input over time, emitting a spike if threshold condition is met
        fc2 = nn.Linear(number_hidden, number_outputs) # Applies linear transformation to output spikes of lif1
        lif2 = snn.Leaky(beta = beta) # Another spiking neuron, integrating the weighted spikes over time

    def forward(self, x):
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
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc * 100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc * 100:.2f}%")

def test_printer():
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
    input_ = df_scaled.iloc[:, :-3].values # Select every column except last three columns of dataframe
    output_ = df_scaled.iloc[:, -3:].values # Select only last three columns of dataframe

    return input_, output_

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

def training_one_iteration(loss, data, targets):
    spk_rec, mem_rec = net(data.view(batch_size, -1))
    print(mem_rec.size())

    # Initailize the total loss value
    loss_val = torch.zeros((1), dtype = dtype, device = device)

    # Sum loss at every step
    for step in range(num_steps):
        loss_val += loss(mem_rec[step], targets)

    print(f"Training loss: {loss_val.item():.3f}")

    print_batch_accuracy(data, targets, train = True)
    # clear previously stored gradients
    optimizer.zero_grad()

    # calculate the gradients
    loss_val.backward()

    # weight update
    optimizer.step()

# Driver code
if __name__ == "__main__":
    TRAINING_PATH = 'training\\'
    TEST_PATH = 'test\\'
    DROP_COLUMNS = ['Unnamed: 0', 'Time Distribution', 'Size Distribution', 'Link Exists', 'Avg Packet Length', 
                'Avg Utilization First', 'Avg Packet Loss Rate', 'Avg Port Occupancy', 'Avg Packet Length First']
    input_train, output_train = process_dataframe(TRAINING_PATH, DROP_COLUMNS)
    print("Input training tensor: " + str(input_train.size()))
    print("Output training tensor: " + str(output_train.size()))
    input_test, output_test = process_dataframe(TEST_PATH, DROP_COLUMNS)
    print("Input test tensor: " + str(input_test.size()))
    print("Output test tensor: " + str(output_test.size()))

    batch_size = 128

    # Passing numpy array to to DataLoader
    train = TensorDataset(input_train, output_train)
    test = TensorDataset(input_test, output_test)
    train_loader = DataLoader(dataset = train, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test, batch_size = batch_size, shuffle = True)

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load network onto CUDA if available
    net = SpikingNeuralNetwork().to(device)
    loss = nn.CrossEntropyLoss() # Softmax of output layer, generate loss at output
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-4, betas = (0.9, 0.999))

    # One iteration of training
    data, targets = next(iter(train_loader))
    data = data.to(device)
    targets = targets.to(device)

    training_one_iteration(loss, data, targets)