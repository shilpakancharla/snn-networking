import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import utils
from snntorch import spikegen
from sklearn import preprocessing
from snntorch import spikeplot as splt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
    le_time = preprocessing.LabelEncoder()
    df_raw['Time Distribution Encoded'] = le_time.fit_transform(df_raw['Time Distribution'])
    df_raw = df_raw.drop(['Time Distribution'], axis = 1)
    
    # Encode the Size Distribution column
    le_size = preprocessing.LabelEncoder()
    df_raw['Size Distribution Encoded'] = le_size.fit_transform(df_raw['Size Distribution'])
    df_raw = df_raw.drop(['Size Distribution'], axis = 1)

    # Scale the input data using StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_raw), columns = df_raw.columns)
    
    # Put dataframe into tensor structure for feeding into spiking neural network
    torch_input_tensor = torch.tensor(df_scaled.iloc[:, :-3].values) # Select every column except last three columns of dataframe
    torch_output_tensor = torch.tensor(df_scaled.iloc[:, -3:].values) # Select only last three columns of dataframe
    print(torch_input_tensor.size())
    print(torch_output_tensor.size())
    return 

def create_snn():
    # Layer parameters
    number_inputs = 23
    number_hidden = 1000
    number_outputs = 3
    beta = 0.99

    # Initialize layers
    fc1 = nn.Linear(number_inputs, number_hidden)
    lif1 = snn.Leaky(beta = beta)
    fc2 = nn.Linear(number_hidden, number_outputs)
    lif2 = snn.Leaky(beta = beta)

    # Initialize hidden states
    mem1 = lif1.init_leaky()
    mem2 = lif2.init_leaky()

    # Record outputs
    mem2_rec = []
    spk1_rec = []
    spk2_rec = []

# Driver code
if __name__ == "__main__":
    csv_filepath = 'tabular_data/25/results_25_400-2000_325_349.csv'
    df = process_dataframe(csv_filepath)

