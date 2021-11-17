import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import utils, spikegen
from snntorch import spikeplot as splt
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class SpikingNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

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

"""
    Process all .csv files to create a dataframe that will be encoded and standardized. Convert this structure to a tensor
    and shape it into the required format for the SNN.

    @param csv_filepath: filepath of where all the .csv files are
    @return tensor of input data in appropriate SNN dimensions
    @return tensor of output data in appropriate SNN dimensions
"""
def process_dataframe(csv_filepath):
    df = pd.read_csv(csv_filepath)
    # Outputs of interest: Avg Packet Loss, Avg Utilization, Max Queue Occupancy
    df_raw = df.drop(['Link Exists', 
                        'Avg Packet Length', 
                        'Avg Utilization First', 
                        'Avg Packet Loss Rate', 
                        'Avg Port Occupancy',
                        'Avg Packet Length First'], axis = 1)
    
    # Encode the Time Distribution column
    le_time = LabelEncoder()
    df_raw['Time Distribution Encoded'] = le_time.fit_transform(df_raw['Time Distribution'])
    df_raw = df_raw.drop(['Time Distribution'], axis = 1)
    
    # Encode the Size Distribution column
    le_size = LabelEncoder()
    df_raw['Size Distribution Encoded'] = le_size.fit_transform(df_raw['Size Distribution'])
    df_raw = df_raw.drop(['Size Distribution'], axis = 1)

    # Scale the input data using StandardScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_raw), columns = df_raw.columns)
    
    # Convert dataframe to tensor
    tensor_obj = torch.tensor(df_scaled.values)

    # Convert dataset to spiking dataset using rate coding
    spike_data_rate = rate_coding(tensor_obj, 25, 200)
        
    # Put dataframe into tensor structure for feeding into spiking neural network
    #torch_input_tensor = torch.tensor(df_scaled.iloc[:, :-3].values) # Select every column except last three columns of dataframe
    #torch_output_tensor = torch.tensor(df_scaled.iloc[:, -3:].values) # Select only last three columns of dataframe

    return 

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

def train_validation_test_split(df):
    return

# Driver code
if __name__ == "__main__":
    csv_filepath = 'tabular_data/25/results_25_400-2000_325_349.csv'
    df = process_dataframe(csv_filepath)

    # Layer parameters
    number_inputs = 23
    number_hidden = 1000
    number_outputs = 3
    beta = 0.99