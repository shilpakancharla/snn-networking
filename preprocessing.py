import os 
import tarfile 
import networkx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class NetworkInput:
    def __init__(self, topology_size, input_filepath, traffic_filepath, link_filepath, graph_filepath, sim_filepath):
        self.topology_size = topology_size
        self.input_filepath = input_filepath
        self.traffic_filepath = traffic_filepath
        self.link_filepath = link_filepath
        self.graph_filepath = graph_filepath
        self.sim_filepath = sim_filepath
        
        # Data extraction and collection
        self.simulation_numbers, routing_matrices = self.process_input_file(self.input_filepath)
        #self.routes = self.process_routing_matrix(routing_matrices)
        self.list_of_traffic_measurements = self.get_traffic_metrics(self.traffic_filepath)
        self.global_packets_list, self.global_losses_list, self.global_delay_list, self.metrics_list = self.get_simulation_metrics(self.sim_filepath)
        self.topology_object = self.graph_process(graph_filepath)
        self.port_statistics_list = self.get_link_usage_metrics(self.link_filepath)

    def process_routing_matrix(self, routing_matrices):
        for i in range(0, self.topology_size, 1):
            for j in range(0, self.topology_size, 1):
                for k in routing_matrices[i][j]:
                    print(k)
    """

    """
    def write_to_csv(self):
        # Processing data from simulationResults.txt
        frames_metrics = []
        for i in range(0, len(self.metrics_list)):
            df_temp = pd.DataFrame(self.metrics_list[i])
            frames_metrics.append(df_temp)
        # Dataframe of simulationResults.txt
        metrics_result_df = pd.concat(frames_metrics, ignore_index = True)

        # Processing data from traffic.txt

        print('Traffic measurements: ', len(self.list_of_traffic_measurements))
        print('Port statistics: ', len(self.port_statistics_list))
    
    """
        Return the traffic metrics as a dictionary with the maximum average lambda value as the key and the
        metrics for each simulation as values. 

        @param filepath: traffic file for simulation
        @return list of traffic measurements with description of time and size distribution parameters
    """
    def get_traffic_metrics(self, filepath):
        traffic_file = open(filepath)
        max_avg_lambda_list = []
        traffic_measurements = []
        for line in traffic_file:
            traffic_tokens = line.split()
            max_avg_lambda = None
            for token in traffic_tokens:
                traffic_data = token.split(';')
                max_avg_lambda, rest_of_token = self.get_max_avg_lambda(traffic_data[0])
                first_tokens = rest_of_token.split(',')
                tokens = []
                for t in first_tokens:
                    tokens.append(t)
                traffic_measurements.append(tokens)
                modified_tokens = traffic_data[1:]
                for m in modified_tokens:
                    rest_to_add = m.split(',')
                    tokens_ = []
                    for r in rest_to_add:
                        if r == None:
                            continue
                        else:
                            tokens_.append(r)
                    max_avg_lambda_list.append(max_avg_lambda)
                    traffic_measurements.append(tokens_)

        traffic_file.close() # Close file once done processing
        traffic_measurements_modified = [x for x in traffic_measurements if x != ['']] # Remove empty lists from list

        # Match up the max avg lambda value with the tokens using list comprehension
        list_dict_with_max_lambda = []
        for m, t in zip(max_avg_lambda_list, traffic_measurements_modified):
            match_dict = dict()
            t_parameterized = self.get_time_size_distribution_parameters(t)
            match_dict[m] = t_parameterized 
            list_dict_with_max_lambda.append(match_dict)

        return list_dict_with_max_lambda

    """
        Get and process the time and size distribution parameters.

        @param traffic_measurement: traffic measurement to process
        @return dictionary of traffic measurement with appropriate time and size distribution parameters
    """
    def get_time_size_distribution_parameters(self, traffic_measurement):
        traffic_dictionary = dict()
        for i in range(len(traffic_measurement)):
            offset = self.create_traffic_time_distribution(traffic_measurement, traffic_dictionary)
            if offset != -1: # Go through this loop if we have a valid offset
                self.create_traffic_size_distribution(traffic_measurement, offset, traffic_dictionary)
        return traffic_dictionary

    """
        Fill out dictionary with traffic time distribution metrics and return the offset of where to read size distribution parameters.

        @param traffic_metrics: list of all the flow traffic metrics to be processed
        @param traffic_dictionary: dictionary to fill out with the time distribution information
        @return number of elements read from the list of parameters data
    """
    def create_traffic_time_distribution(self, traffic_metrics, traffic_dictionary):
        if int(traffic_metrics[0]) == 0:
            traffic_dictionary['Time Distribution'] = 'Exponential'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Average Packet Lambda'] = float(traffic_metrics[2])
            parameters['Exponential Max Factor'] = float(traffic_metrics[3])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 4
        elif int(traffic_metrics[0]) == 1:
            traffic_dictionary['Time Distribution'] = 'Deterministic'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Average Packet Lambda'] = float(traffic_metrics[2])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 3
        elif int(traffic_metrics[0]) == 2:
            traffic_dictionary['Time Distribution'] = 'Uniform'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Min Packet Lambda'] = float(traffic_metrics[2])
            parameters['Max Packet Lambda'] = float(traffic_metrics[3])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 4
        elif int(traffic_metrics[0]) == 3:
            traffic_dictionary['Time Distribution'] = 'Normal'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Average Packet Lambda'] = float(traffic_metrics[2])
            parameters['Standard Deviation'] = float(traffic_metrics[3])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 4
        elif int(traffic_metrics[0]) == 4:
            traffic_dictionary['Time Distribution'] = 'OnOff'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Packets Lambda On'] = float(traffic_metrics[2])
            parameters['Average Time Off'] = float(traffic_metrics[3])
            parameters['Average Time On'] = float(traffic_metrics[4])
            parameters['Exponential Max Factor'] = float(traffic_metrics[5])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 6
        elif int(traffic_metrics[0]) == 5:
            traffic_dictionary['Time Distribution'] = 'PPBP'
            parameters = dict()
            parameters['Equivalent Lambda'] = float(traffic_metrics[1])
            parameters['Burst Gen Lambda'] = float(traffic_metrics[2])
            parameters['Bit Rate'] = float(traffic_metrics[3])
            parameters['Pare to Min Size'] = float(traffic_metrics[4])
            parameters['Pare to Max Size'] = float(traffic_metrics[5])
            parameters['Pare to Alpha'] = float(traffic_metrics[6])
            parameters['Exponential Max Factor'] = float(traffic_metrics[7])
            traffic_dictionary['Time Distribution Parameters'] = parameters
            return 8
        else:
            return -1

    """
        Retrieve the size distribution parameters of the traffic data. 

        @param traffic_metrics: list of all the flow traffic metrics to be processed
        @param offset: number of elements read from the list of parameters data
        @param traffic_dictionary: dictionary to fill out with the time distribution information
        @return 0 if successful iteration, or -1 otherwise
    """
    def create_traffic_size_distribution(self, traffic_metrics, offset, traffic_dictionary):
        if int(traffic_metrics[offset]) == 0:
            traffic_dictionary['Size Distribution'] = 'Deterministic'
            parameters = dict()
            parameters['Average Packet Size'] = float(traffic_metrics[offset + 1])
            traffic_dictionary['Size Distribution Parameters'] = parameters
        elif int(traffic_metrics[offset]) == 1:
            traffic_dictionary['Size Distribution'] = 'Uniform'
            parameters = dict()
            parameters['Average Packet Size'] = float(traffic_metrics[offset + 1])
            parameters['Min Size'] = float(traffic_metrics[offset + 2])
            parameters['Max Size'] = float(traffic_metrics[offset + 3])
            traffic_dictionary['Size Distribution Parameters'] = parameters
        elif int(traffic_metrics[offset]) == 2:
            traffic_dictionary['Size Distribution'] = 'Binomial'
            parameters = dict()
            parameters['Average Packet Size'] = float(traffic_metrics[offset + 1])
            parameters['Packet Size 1'] = float(traffic_metrics[offset + 2])
            parameters['Packet Size 2'] = float(traffic_metrics[offset + 3])
            traffic_dictionary['Size Distribution Parameters'] = parameters
        elif int(traffic_metrics[offset]) == 3:
            traffic_dictionary['Size Distribution'] = 'Generic'
            parameters = dict()
            parameters['Average Packet Size'] = float(traffic_metrics[offset + 1])
            parameters['Number of Candidates'] = float(traffic_metrics[offset + 2])
            for i in range(0, int(traffic_metrics[offset + 2]) * 2, 2):
                parameters['Size %d' % (i / 2)] = float(traffic_metrics[offset + 3 + i])
                parameters['Prob %d' % (i / 2)] = float(traffic_dictionary[offset + 4 + i])
            traffic_dictionary['Size Distribution Parameters'] = parameters
        else:
            return -1
        return 0
        
    """
        Get the maximum average lambda value and the string without it.

        @param token: original string containing the maximum average lambda value and the rest of the string
        @return maximum average lambda value
        @return rest of string without maximum average lambda value
    """
    def get_max_avg_lambda(self, token):
        bar_index = token.find('|')
        return token[0:bar_index], token[bar_index:].replace('|', '')

    """
        Extract information from the input file containing information about simulation number, graph files, and routing files.
        
        @param filepath: input file path
        @return simulation numbers
        @return list of MatrixPath objects showing routings, if they exist
    """
    def process_input_file(self, filepath):
        input_file = open(filepath)
        sim_numbers = []
        graph_files = []
        routing_files = []
        net_size = filepath.replace('training_data\gnnet-ch21-dataset-train\\', '')[0:2]
        for line in input_file:
            input_data = line.split(';')
            sim_numbers.append(input_data[0])
            filepath_stem = 'training_data\gnnet-ch21-dataset-train\\'
            graph_file = filepath_stem + net_size + '\graphs\\' + input_data[1]
            graph_files.append(graph_file)
            routing_file = (filepath_stem + net_size + '\\routings\\' + input_data[2]).rstrip()
            routing_files.append(routing_file)
        
        input_file.close() # Close file once done processing

        # Create routing matrix
        routing_matrices = []
        for graph, route in zip(graph_files, routing_files):
            routing_matrix = self.create_routing_matrix(int(net_size), route)
            routing_matrices.append(routing_matrix)
        
        return sim_numbers, routing_matrices 

    """
        Creating a routing object to show an n x n matrix and how each cell [i, j] contains the path to go from node i
        to node j. If 'None' value is present, no route is available. 

        @param net_size: size of topology network
        @param routing_file: file regarding information about routing configuration
        @return MatrixPath object that shows routes between different nodes, if present
    """
    def create_routing_matrix(self, net_size, routing_file):
        MatrixPath = np.empty((net_size, net_size), dtype = object)
        with open(routing_file) as rf:
            for line in rf:
                nodes = line.split(';')
                nodes = list(map(int, nodes))
                MatrixPath[nodes[0], nodes[-1]] = nodes
        return (MatrixPath)

    """
        Create and return a graph readable structure for a specified GML markup.

        @param filepath: directory of GML markup files for a topology size
        @return graph structure with information about nodes and edges
    """
    def graph_process(self, filepath):
        graphs_dictionary = dict()
        for topology_file in os.listdir(filepath):
            G = networkx.read_gml(filepath + '/' + topology_file, destringizer = int)
            #networkx.draw(G, with_labels = True, font_weight = 'bold')
            graphs_dictionary[topology_file] = G
            # Nodes of graph
            nodes = G.nodes
            # Topology edges
            edges = G.edges # Returns list of tuples describing topology edges
            # Information parameters related to a node
            for i in nodes:
                graphs_dictionary[i] = nodes[i]
            # Information about link parameters
            for i in edges: # (src node ID, dest node ID, link ID)
                graphs_dictionary[i] = G[i[0]][i[1]][0]
        return graphs_dictionary
    
    """
        Given a simulation metrics file, extract information about the global packets, global losses, global delays, and
        the metrics list for each instance. 

        @param filepath: simulation file of network topology
        @return list of global packets measurements
        @return list of global losses measurements
        @return list of global delay measurements
        @return list of dictionaries of metrics (performance matrix)
    """
    def get_simulation_metrics(self, filepath):
        sim_file = open(filepath)
        global_packets_list = []
        global_losses_list = []
        global_delay_list = []
        metrics_list = []
        for line in sim_file:
            measurement = line.split()
            measurement_tokens = measurement[0].split(',')
            global_packets = measurement_tokens[0] # Get global packets value
            global_packets_list.append(global_packets)
            measurement_tokens.remove(global_packets) # Remove value once finished
            global_losses = measurement_tokens[0] # Get global losses value
            global_losses_list.append(global_losses)
            measurement_tokens.remove(global_losses) # Remove value once finished
            global_delay_temp = measurement_tokens[0] # Get global delay value and modify
            bar_index = global_delay_temp.find('|')
            global_delay = global_delay_temp[0:bar_index]
            global_delay_list.append(global_delay)

            metrics_list_ = self.get_traffic_measurements(global_packets, global_losses, global_delay, measurement_tokens) # Get the rest of the list metrics
            metrics_list.append(metrics_list_)
        
        sim_file.close() # Close the file once done processing
        return global_packets_list, global_losses_list, global_delay_list, metrics_list

    """
        Extracting the individual list metrics and appending them all to one array. 

        @param global_delay: used to remove before iterating through the measurement tokens
        @param measurement_tokens: array of metrics that are separated by '|' and ';'
        @return list of dictionaries of metrics
    """
    def get_traffic_measurements(self, global_packet, global_loss, global_delay, measurement_tokens):
        metrics_list = []
        modified_measurements = self.modify_tokens(measurement_tokens)  
        if global_delay in modified_measurements:
            modified_measurements.remove(global_delay)  
        # Get the rest of the simulation tokens
        counter = 0
        while (counter < len(modified_measurements)):
            metric_aggregated_dictionary = dict() # Re-initialize dictionary to hold the metrics
            for i in range(0, 11, len(modified_measurements)):
                # Add global values
                metric_aggregated_dictionary['Global Packet'] = global_packet
                metric_aggregated_dictionary['Global Loss'] = global_loss
                metric_aggregated_dictionary['Global Delay'] = global_delay
                # Bandwidth
                bandwidth = modified_measurements[i]
                metric_aggregated_dictionary['Average Bandwidth'] = bandwidth
                # Number of packets transmitted
                number_packets_transmitted = modified_measurements[i + 1]
                metric_aggregated_dictionary['Packets Transmitted'] = number_packets_transmitted
                # Number of packets dropped
                number_packets_dropped = modified_measurements[i + 2]
                metric_aggregated_dictionary['Packets Dropped'] = number_packets_dropped 
                # Average per-packet delay
                avg_delay = modified_measurements[i + 3]
                metric_aggregated_dictionary['Average Per-Packet Delay'] = avg_delay
                # Neperian logarithm of per-packet delay
                neperian_logarithm = modified_measurements[i + 4]
                metric_aggregated_dictionary['Neperian Logarithm'] = neperian_logarithm
                # Percentile 10 of per-packet delay
                percentile_10 = modified_measurements[i + 5]
                metric_aggregated_dictionary['Percentile 10'] = percentile_10 
                # Percentile 20 of per-packet delay
                percentile_20 = modified_measurements[i + 6]
                metric_aggregated_dictionary['Percentile 20'] = percentile_20 
                # Percentile 50 of per-packet delay
                percentile_50 = modified_measurements[i + 7]
                metric_aggregated_dictionary['Percentile 50'] = percentile_50
                # Percentile 80 of per-packet delay
                percentile_80 = modified_measurements[i + 8]
                metric_aggregated_dictionary['Percentile 80'] = percentile_80 
                # Percentile 90 of per-packet delay
                percentile_90 = modified_measurements[i + 9]
                metric_aggregated_dictionary['Percentile 90'] = percentile_90 
                # Variance of per-packet delay (jitter)
                variance_delay = modified_measurements[i + 10]
                metric_aggregated_dictionary['Jitter'] = variance_delay
            
            counter = counter + 1 # Increment counter
            metrics_list.append(metric_aggregated_dictionary) # Append to master list of metrics

        return metrics_list

    """
        Modify the measurement tokens list so that are no '|' or ';' present, but all tokens are separated and present. 

        @param measurement_tokens: array of metrics that are separated by '|' and ';'
        @return modified list in which each token is present in the list
    """
    def modify_tokens(self, measurement_tokens):
        modified_measurements = []
        # Modify measurements so they can be read and organized
        for token in measurement_tokens:
            if '|' in token:
                new_token = token.split('|')
                for t in new_token:
                    modified_measurements.append(t)
            elif ';' in token:
                new_token = token.split(';')
                for t in new_token:
                    modified_measurements.append(t)
            else:
                modified_measurements.append(token)
        return modified_measurements
    """
        Given a link usage file, extract information about various metrics present. If a link does not exist, populate the 
        list of metrics with all -1. 

        @param filepath: link usage file of source-destination pair
        @return list of port statistics by each measure
    """
    def get_link_usage_metrics(self, filepath):
        link_file = open(filepath)
        port_statistics_list = []
        for line in link_file:
            port_statistics = line.split(';')
            for token in port_statistics:
                port_statistics_individual = []
                if token == '-1':
                    # Link does not exist
                    link_exists = False
                    port_statistics_individual.append(link_exists)
                    for i in range(8):
                        port_statistics_individual.append(-1)
                    port_statistics_list.append(port_statistics_individual)
                elif token == '\n': # Go to next line at end of line
                    break
                else:
                    tokens = token.split(',')
                    # The link exists
                    link_exists = True
                    port_statistics_individual.append(link_exists)
                    # Average utilization of the outgoing port
                    avg_utilization = tokens[0]
                    port_statistics_individual.append(avg_utilization)
                    # Average packet loss rate in the outgoing port
                    avg_packet_loss = tokens[1]
                    port_statistics_individual.append(avg_packet_loss)
                    # Average packet length of the packets transmitted through the outgoing port
                    avg_packet_length = tokens[2]
                    port_statistics_individual.append(avg_packet_length)
                    # Average utilization of the first queue of the outgoing port
                    avg_utilization_first = tokens[3]
                    port_statistics_individual.append(avg_utilization_first)
                    # Average packet loss rate in the first queue of the outgoing port
                    avg_packet_loss_rate = tokens[4]
                    port_statistics_individual.append(avg_packet_loss_rate)
                    # Average port occupancy (service and waiting queue) of the first queue of the outgoing port
                    avg_port_occupancy = tokens[5]
                    port_statistics_individual.append(avg_port_occupancy)
                    # Maximum queue occupancy of the first queue of the outgoing port
                    max_queue_occupancy = tokens[6]
                    port_statistics_individual.append(max_queue_occupancy)
                    # Average packet length of the packets transmitted through the first queue of the outgoing port
                    avg_packet_length_first = tokens[7]
                    port_statistics_individual.append(avg_packet_length_first)
                    port_statistics_list.append(port_statistics_individual)

        link_file.close() # Close the file once done processing
        return port_statistics_list
    
    """
        Function for plotting the types of time and size distributions for a particular network topology.

        @param list_of_traffic_measurements: list of dictionaries of time and size distributions of traffic data
        @param topology_size: number of nodes in network
    """
    def plot_size_dist_type(self, list_of_traffic_measurements, topology_size):
        time_distribution_type = []
        size_distribution_type = []
        for traffic_measurement in list_of_traffic_measurements:
            for key in traffic_measurement:
                if key == 'Time Distribution':
                    time_distribution_type.append(traffic_measurement['Time Distribution'])
                elif key == 'Size Distribution':
                    size_distribution_type.append(traffic_measurement['Size Distribution'])
                else:
                    continue
        time_keys = Counter(time_distribution_type).keys()
        time_values = Counter(time_distribution_type).values()
        size_keys = Counter(size_distribution_type).keys()
        size_values = Counter(size_distribution_type).values()

        # Create bar graph for time distribution
        time_x_pos = [i for i, _ in enumerate(time_keys)]
        plt.bar(time_x_pos, time_values, color = 'aquamarine')
        plt.xlabel("Time Distribution Type")
        plt.ylabel("Count")
        plt.title("Time Distributions for Topology Size " + str(topology_size))
        plt.xticks(time_x_pos, time_keys)
        #plt.show()
        plt.savefig(f'plots/time_distribution_{topology_size}.png')
        
        # Create bar graph for size distribution
        size_x_pos = [i for i, _ in enumerate(size_keys)]
        plt.bar(size_x_pos, size_values, color = 'deeppink')
        plt.xlabel("Size Distribution Type")
        plt.ylabel("Count")
        plt.title("Size Distributions for Topology Size " + str(topology_size))
        plt.xticks(size_x_pos, size_keys)
        #plt.show()
        plt.savefig(f'plots/size_distribution_{topology_size}.png')

    """
        Creates plots of various time distribution characters of traffic metrics.

        @param list_of_traffic_measurements: list of dictionaries of time and size distributions of traffic data
        @param topology_size: number of nodes in network
    """
    def plot_traffic_time_characteristics(self, list_of_traffic_measurements, topology_size):
        equivalent_lambda_exp = []
        avg_packets_lambda_exp = []
        exp_max_factor_exp = []
        equivalent_lambda_det = []
        avg_packets_lambda_det = []
        equivalent_lambda_uniform = []
        min_packet_lambda = []
        max_packet_lambda = []
        equivalent_lambda_normal = []
        avg_packet_lambda_normal = []
        std_dev_normal = []
        equivalent_lambda_onoff = []
        packets_lambda_on = []
        avg_time_off = []
        avg_time_on = []
        exp_max_factor_onoff = []
        equivalent_lambda_ppbp = []
        burst_gen_lambda = []
        bit_rate = []
        pare_min_size = []
        pare_max_size = []
        pare_alpha = []
        exp_max_factor_ppbp = []
        for traffic_measurement in list_of_traffic_measurements:
            if traffic_measurement['Time Distribution'] == 'Exponential':
                # Collect exponential distribution characteristics
                equivalent_lambda_exp.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                avg_packets_lambda_exp.append(traffic_measurement['Time Distribution Parameters']['Average Packet Lambda'])
                exp_max_factor_exp.append(traffic_measurement['Time Distribution Parameters']['Exponential Max Factor'])
            elif traffic_measurement['Time Distribution'] == 'Deterministic':
                # Collect deterministic distribution characteristics
                equivalent_lambda_det.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                avg_packets_lambda_det.append(traffic_measurement['Time Distribution Parameters']['Average Packet Lambda'])
            elif traffic_measurement['Time Distribution'] == 'Uniform':
                # Collect uniform distribution characteristics
                equivalent_lambda_uniform.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                min_packet_lambda.append(traffic_measurement['Time Distribution Parameters']['Min Packet Lambda'])
                max_packet_lambda.append(traffic_measurement['Time Distribution Parameters']['Max Packet Lambda'])
            elif traffic_measurement['Time Distribution'] == 'Normal':
                # Collect normal distribution characteristics
                equivalent_lambda_normal.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                avg_packet_lambda_normal.append(traffic_measurement['Time Distribution Parameters']['Average Packet Lambda'])
                std_dev_normal.append(traffic_measurement['Time Distribution Parameters']['Standard Deviation'])
            elif traffic_measurement['Time Distribution'] == 'OnOff':
                # Collect on-off distribution characteristics
                equivalent_lambda_onoff.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                packets_lambda_on.append(traffic_measurement['Time Distribution Parameters']['Packets Lambda On'])
                avg_time_off.append(traffic_measurement['Time Distribution Parameters']['Average Time Off'])
                avg_time_on.append(traffic_measurement['Time Distribution Parameters']['Average Time On'])
                exp_max_factor_onoff.append(traffic_measurement['Time Distribution Parameters']['Exponential Max Factor'])
            elif traffic_measurement['Time Distribution'] == 'PPBP':
                # Collect PPBP distribution characteristics
                equivalent_lambda_ppbp.append(traffic_measurement['Time Distribution Parameters']['Equivalent Lambda'])
                burst_gen_lambda.append(traffic_measurement['Time Distribution Parameters']['Burst Gen Lambda'])
                bit_rate.append(traffic_measurement['Time Distribution Parameters']['Bit Rate'])
                pare_min_size.append(traffic_measurement['Time Distribution Parameters']['Pare to Min Size'])
                pare_max_size.append(traffic_measurement['Time Distribution Parameters']['Pare to Max Size'])
                pare_alpha.append(traffic_measurement['Time Distribution Parameters']['Pare to Alpha'])
                exp_max_factor_ppbp.append(traffic_measurement['Time Distribution Parameters']['Exponential Max Factor'])
            else:
                continue
        
        # Plot exponential distribution statistics
        if len(equivalent_lambda_exp) != 0:
            plt.boxplot([equivalent_lambda_exp, avg_packets_lambda_exp, exp_max_factor_exp], 
                    vert = True,
                    labels = ['Equivalent Lambda', 'Average Packets Lambda', 'Exponential Max Factor'],
                    meanline = True)
            plt.title("Exponential (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/exp/{topology_size}/time_exponential_dist_statistics_{topology_size}.png')
        
        # Plot deterministic distribution statistics
        if len(equivalent_lambda_det) != 0:
            plt.boxplot([equivalent_lambda_det, avg_packets_lambda_det],
                    vert = False,
                    labels = ['Equivalent Lambda', 'Average Packets Lambda'],
                    meanline = True)
            plt.title("Deterministic (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/det/{topology_size}/time_deterministic_dist_statistics_{topology_size}.png')
        
        # Plot uniform distribution statistics
        if len(equivalent_lambda_uniform) != 0:
            plt.boxplot([equivalent_lambda_uniform, min_packet_lambda, max_packet_lambda],
                    vert = False,
                    labels = ['Equivalent Lambda', 'Min Packet Lambda', 'Max Packet Lambda'],
                    meanline = True)
            plt.title("Uniform (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/uniform/{topology_size}/time_uniform_dist_statistics_{topology_size}.png')
        
        # Plot normal distribution statistics
        if len(equivalent_lambda_normal) != 0:
            plt.boxplot([equivalent_lambda_normal, avg_packet_lambda_normal, std_dev_normal],
                    vert = False,
                    labels = ['Equivalent Lambda', 'Average Packet Lambda', 'Standard Deviation'],
                    meanline = True)
            plt.title("Normal (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/normal/{topology_size}/time_normal_dist_statistics_{topology_size}.png')
        
        # Plot on-off distribution statistics
        if len(equivalent_lambda_onoff) != 0:
            plt.boxplot([equivalent_lambda_onoff, packets_lambda_on, avg_time_off, avg_time_on, exp_max_factor_onoff],
                    vert = False,
                    labels = ['Equivalent Lambda', 'Packets Lambda On', 'Average Time Off', 'Average Time On', 'Exponential Max Factor'],
                    meanline = True)
            plt.title("On-off (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/onoff/{topology_size}/time_onoff_dist_statistics_{topology_size}.png')
        
        # Plot PPBP distribution statistics
        if len(equivalent_lambda_ppbp) != 0:
            plt.boxplot([equivalent_lambda_ppbp, burst_gen_lambda, bit_rate, pare_min_size, pare_max_size, pare_alpha, exp_max_factor_ppbp],
                    vert = False,
                    labels = ['Equivalent Lambda', 'Burst Gen Lambda', 'Bit Rate', 'Pare to Min Size', 'Pare to Max Size', 'Pare to Alpha', 'Exponential Max Factor'],
                    meanline = True)
            plt.title("PPBP (Time) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/time/ppbp/{topology_size}/time_ppbp_dist_statistics_{topology_size}.png')
    """
        Creates plots of various size distribution characters of traffic metrics.

        @param list_of_traffic_measurements: list of dictionaries of time and size distributions of traffic data
        @param topology_size: number of nodes in network
    """
    def plot_traffic_size_characteristics(self, list_of_traffic_measurements):
        avg_packet_size_det = []
        avg_packet_size_uniform = []
        min_size = []
        max_size = []
        avg_packet_size_bi = []
        packet_size_1 = []
        packet_size_2 = []
        avg_packet_size_generic = []
        number_of_candidates = []
        for traffic_measurement in list_of_traffic_measurements:
            if traffic_measurement['Size Distribution'] == 'Deterministic':
                # Collect deterministic distribution characteristics
                avg_packet_size_det.append(traffic_measurement['Size Distribution Parameters']['Average Packet Size'])
            elif traffic_measurement['Size Distribution'] == 'Uniform':
                # Collect uniform distribution characteristics
                avg_packet_size_uniform.append(traffic_measurement['Size Distribution Parameters']['Average Packet Size'])
                min_size.append(traffic_measurement['Size Distribution Parameters']['Min Size'])
                max_size.append(traffic_measurement['Size Distribution Parameters']['Max Size'])
            elif traffic_measurement['Size Distribution'] == 'Binomial':
                # Collect binomial distribution characteristics
                avg_packet_size_bi.append(traffic_measurement['Size Distribution Parameters']['Average Packet Size'])
                packet_size_1.append(traffic_measurement['Size Distribution Parameters']['Packet Size 1'])
                packet_size_2.append(traffic_measurement['Size Distribution Parameters']['Packet Size 2'])
            elif traffic_measurement['Size Distribution'] == 'Generic':
                # Collect generic distribution characteristics
                avg_packet_size_generic.append(traffic_measurement['Size Distribution Parameters']['Average Packet Size'])
                number_of_candidates.append(traffic_measurement['Size Distribution Parameters']['Number of Candidates'])
            else:
                continue

        # Plot deterministic distribution statistics
        if len(avg_packet_size_det) != 0:
            plt.boxplot(avg_packet_size_det,
                    vert = False,
                    labels = 'Average Packet Size',
                    meanline = True)
            plt.title("Deterministic (Size) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/size/det/{topology_size}/size_deterministic_dist_statistics_{topology_size}.png')
        
        # Plot uniform distribution statistics
        if len(avg_packet_size_uniform) != 0:
            plt.boxplot([avg_packet_size_uniform, min_size, max_size],
                    vert = False,
                    labels = ['Average Packet Size', 'Min Size', 'Max Size'],
                    meanline = True)
            plt.title("Uniform (Size) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/size/uniform/{topology_size}/size_uniform_dist_statistics_{topology_size}.png')
        
        # Plot binomial distribution statistics
        if len(avg_packet_size_bi) != 0:
            plt.boxplot([avg_packet_size_bi, packet_size_1, packet_size_2],
                    vert = False,
                    labels = ['Average Packet Size', 'Packet Size 1', 'Packet Size 2'],
                    meanline = True)
            plt.title("Binomial (Size) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/size/bi/{topology_size}/size_binomial_dist_statistics_{topology_size}.png')
        
        # Plot generic distribution statistics
        if len(avg_packet_size_generic) != 0:
            plt.boxplot([avg_packet_size_generic, number_of_candidates],
                    vert = False,
                    labels = ['Average Packet Size', 'Number of Candidates'],
                    meanline = True)
            plt.title("Generic (Size) Statistics for Topology Size " + str(topology_size))
            plt.savefig(f'plots/size/gen/{topology_size}/size_generic_dist_statistics_{topology_size}.png')

"""
    Extract contents of tar files downloaded.

    @param filename: path and name of the tar file (.tar)
    @param path: where data should be located
"""
def extract(filename, path):
    if filename.endswith("tar.gz"):
        tar = tarfile.open(filename, "r:gz")
        tar.extractall(path = path)
        print("Path 1: " + filename + " unzipped.")
        tar.close()
        return
    elif filename.endswith("tar"):
        tar = tarfile.open(filename, "r:")
        tar.extractall(path = path)
        print("Path 2: " + filename + " unzipped.")
        tar.close()
        return
    else:
        print("Not a tar file.")
        return

"""
    Within a directory, extract all the contents within all the tar files in that directory. 
    Print the progress of the extraction.

    @param main_filepath: file path that contains all the tar files
"""
def extract_all_in_filepath(main_filepath):
    files = os.listdir(main_filepath)
    files_length = len(files)
    extraction_progress = 0
    for tar_file in os.listdir(main_filepath):
        if tar_file.endswith('.gz'): # Tar files only 
            # All the names end with 'tar.gz' in this path
            new_directory_path = main_filepath + tar_file.replace('tar.gz', '')
            os.mkdir(new_directory_path)
            extract(main_filepath + tar_file, new_directory_path)
            extraction_progress = extraction_progress + 1
            print("Unzipped " + str(extraction_progress) + " out of " + str(files_length) + " files.")

# Driver code
TRAINING_PATH = 'training_data\gnnet-ch21-dataset-train\\'
INPUT_FILE = TRAINING_PATH + '25\\results_25_400-2000_0_24\\results_25_400-2000_0_24\input_files.txt'
TRAFFIC_FILE = TRAINING_PATH + '25\\results_25_400-2000_0_24\\results_25_400-2000_0_24\\traffic.txt'
LINK_FILE = TRAINING_PATH + '25\\results_25_400-2000_0_24\\results_25_400-2000_0_24\linkUsage.txt'
GRAPH_FILES = TRAINING_PATH + '25\\graphs\\'
SIM_FILE = TRAINING_PATH + '25\\results_25_400-2000_0_24\\results_25_400-2000_0_24\simulationResults.txt'
dataset = NetworkInput(25, INPUT_FILE, TRAFFIC_FILE, LINK_FILE, GRAPH_FILES, SIM_FILE)
dataset.write_to_csv()