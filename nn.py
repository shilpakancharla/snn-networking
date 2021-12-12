import torch
import numpy as np
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import utils, spikegen
from snntorch import spikeplot as splt
from torch.utils.data import DataLoader, TensorDataset

"""
    Author: Shilpa Kancharla
    Last Modified: December 12, 2021
"""

class NeuralNetwork(nn.Module):

    """
        Parameters in NeuralNetwork class:
        
        1. number_inputs: Number of inputs to the SNN.
        2. number_hidden: Number of hidden layers.
        3. number_outputs: Number of output classes.
    """

    def __init__(self, number_inputs, number_hidden, number_outputs):
        super().__init__()
        self.number_inputs = number_inputs
        self.number_hidden = number_hidden
        self.number_outputs = number_outputs
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.number_inputs, self.number_hidden),
            nn.ReLU(),
            nn.Linear(self.number_hidden, self.number_hidden),
            nn.ReLU(),
            nn.Linear(self.number_hidden, self.number_hidden),
            nn.Dropout(p = 0.2),
            nn.Linear(self.number_hidden, self.number_outputs)
        )

    """
        Forward propagation of neural network.

        @param x: input data
        @return output value
    """
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

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
                y_pred = net(data.view(batch_size, 21))
            except RuntimeError:
                print("Hit RuntimeError.")
                return loss_history, test_loss_history, acc_history, test_acc_history # Return values to this point

            # Initialize the loss and sum over time
            loss_val = torch.zeros((1), dtype = dtype, device = device)
            for step in range(num_steps):
                loss_val += loss_function(y_pred, targets.float().flatten().to(device).unsqueeze(1))

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
                    y_pred_test = net(test_data.view(batch_size, 21))
                except RuntimeError:
                    print("Hit RuntimeError.")
                    return loss_history, test_loss_history, acc_history, test_acc_history

                # Test set loss
                test_loss = torch.zeros((1), dtype = dtype, device = device)
                for step in range(num_steps):
                    test_loss += loss_function(y_pred_test, test_targets.float().flatten().to(device).unsqueeze(1))
                test_loss_history.append(test_loss.item())

                # Print train/test loss and accuracy
                acc_value, test_acc_value = train_printer(epoch, iter_counter, counter, loss_history, 
                                        data, targets, test_data, test_targets)
                counter = counter + 1
                iter_counter = iter_counter + 1
                acc_history[iter_counter] = acc_value
                test_acc_history[iter_counter] = test_acc_value

                # Break loop if any of these loss criteria are met
                if torch.allclose(test_loss, torch.tensor([0.0009])):
                    count_test_loss = count_test_loss + 1

                if count_test_loss == 3:
                    return loss_history, test_loss_history, acc_history, test_acc_history
    
    return loss_history, test_loss_history, acc_history, test_acc_history

"""
    Calculate the accuracy after each iteration for the train and test sets.

    @param data: feature values
    @param targets: target values
    @param train: Boolean of if we are in train mode or not
    @return accuracy value for the iteration
"""
def print_batch_accuracy(data, targets, train = False):
    output = net(data.view(batch_size, -1))
    idx = output.sum(dim = 0)
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
    plt.savefig('nn_loss_histories.png') 

"""
    Plot the accuracy history.

    @param acc: accuracy history of train data (dictionary)
"""
def plot_accuracy(acc):
    fig = plt.figure(facecolor = 'w', figsize = (20, 10))
    acc_list = acc.items()
    acc_list = sorted(acc_list)
    x_acc, y_acc = zip(*acc_list)
    plt.plot(x_acc, y_acc)
    plt.title("Accuracy Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy %")
    plt.savefig('nn_accuracies.png')  

"""
    Plot the test accuracy history.

    @param test_acc: accuracy history of test data (dictionary)
"""
def plot_test_accuracy(test_acc):
    fig = plt.figure(facecolor = 'w', figsize = (20, 10))
    test_acc_list = test_acc.items()
    test_acc_list = sorted(test_acc_list)
    test_x_acc, test_y_acc = zip(*test_acc_list)
    plt.plot(test_x_acc, test_y_acc)
    plt.title("Test Accuracy Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy %")
    plt.savefig('test_nn_accuracies.png')

# Driver code
if __name__ == "__main__":
    # Load .npy files once you save them
    INPUT_TRAIN = 'npy_files\\input_train.npy'
    OUTPUT_TRAIN = 'npy_files\\output_train.npy'
    INPUT_TEST = 'npy_files\\input_test.npy'
    OUTPUT_TEST = 'npy_files\\output_test.npy'
    features_train_tensor = np.load(INPUT_TRAIN)
    target_train_tensor = np.load(OUTPUT_TRAIN)
    features_test_tensor = np.load(INPUT_TEST)
    target_test_tensor = np.load(OUTPUT_TEST)
    
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
    net = NeuralNetwork(21, 1000, 1).to(device) # Load network onto CUDA if available

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr = 5e-4, betas = (0.9, 0.999))

    #training_one_iteration(train_loader, dtype, device, optimizer)
    loss_history, test_loss_history, acc, test_acc = training_loop(net, train_loader, test_loader, dtype, 
                                                                device, optimizer)

    plot_loss(loss_history, test_loss_history)
    plot_accuracy(acc)
    plot_test_accuracy(test_acc)