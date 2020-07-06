# modifing net es2 the class Line such that it includes an attribute state that can assume values 'free' or
# occupied. This attribute shows if a connection is already occupying that line
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import c

# signal informations

class Signalinformations(object):
    def __init__(self, power, path):
        self._signal_power = power
        self._path = path
        self._noise_power = 0
        self._latency = 0

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise):
        self._noise_power = noise

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    def add_noise(self, noise):
        self.noise_power = self.noise_power + noise

    def add_latency(self, latency):
        self.latency = self.latency+latency

    def next(self):
        self.path = self.path[1:]


class Node(object):
    def __init__(self, node_dict):
        self._label = node_dict['label']
        self._position = node_dict['position']
        self._connected_nodes = node_dict['connected_nodes']
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def propagate(self, signal_information, occupation=False):
        path = signal_information.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            signal_information.next()
            signal_information = line.propagate(signal_information, occupation)
        return signal_information

class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._lenght = line_dict['lenght']
        self._state = 'free'
        self._successive = {}

    @property
    def label(self):
        return self._label

    @property
    def lenght(self):
        return self._lenght

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        state = state.lower().strip()
        if state in ['free', 'occupied']:
            self._state = state
        else:
            print('ERROR: line state not recognized. Value:', state)

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def latency_generation(self):
        latency = self.lenght / ((3*10^8) * (2 / 3))
        return latency

    def noise_generation(self, signal_power):
        noise = signal_power / (2 * self.lenght)
        return noise

    def propagate(self, signal_information, occupation=False):
        #update latency
        latency = self.latency_generation()
        signal_information.add_latency(latency)

         #update noise
        signal_power = signal_information.signal_power
        noise = self.noise_generation(signal_power)
        signal_information.add_noise(noise)

        #update line state
        if occupation:
            self.state = 'occupied'

        node = self.successive[signal_information.path[0]]
        signal_information = node.propagate(signal_information, occupation)

        return signal_information


class Network(object):
    def __init__(self, json_path):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None

        for node_label in node_json:  #create node instance
            node_dict = node_json[node_label]
            node_dict['label'] = node_label
            node = Node(node_dict)
            self._nodes[node_label] = node

            for connected_node_label in node_dict['connected_nodes']:
                line_dict = {}
                line_label = node_label+connected_node_label
                line_dict['label'] = line_label
                node_position = np.array(node_json[node_label]['position'])
                connected_node_position = np.array(node_json[connected_node_label]['position'])
                line_dict['lenght'] = np.sqrt(np.sum((node_position-connected_node_position)**2))
                line = Line(line_dict)
                self._lines[line_label] = line

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    def draw(self):
        nodes = self.nodes
        for node_label in nodes:
            n0 = nodes[node_label]
            x0 = n0.position[0]
            y0 = n0.position[1]
            plt.plot(x0, y0, 'go', markersize=10)
            plt.text(x0+20, y0+20, node_label)
            for connected_node_label in n0.connected_nodes:
                n1 = nodes[connected_node_label]
                x1 = n1.position[0]
                y1 = n1.position[1]
                plt.plot([x0, x1], [y0, y1], 'b')
        plt.title('network')
        plt.show()

    def find_paths(self, label1, label2):
        cross_nodes = []
        for key in self.nodes.keys():
            if (key != label1) & (key != label2):
                cross_nodes.append(key)
        cross_lines = self.lines.keys()
        inner_paths = {}
        inner_paths['0'] = label1
        for i in range(len(cross_nodes)+1):
            inner_paths[str(i+1)] = []
            for inner_path in inner_paths[str(i)]:
                for cross_node in cross_nodes:
                    if (inner_path[-1]+cross_node in cross_lines) & (cross_node not in inner_path):
                        inner_paths[str(i+1)] = inner_paths[str(i+1)]+[inner_path+cross_node]
        paths = []
        for i in range(len(cross_nodes)+1):
            for path in inner_paths[str(i)]:
                if path[-1]+label2 in cross_lines:
                    paths.append(path + label2)
        return paths

    def connect(self):
        nodes_dict = self.nodes
        lines_dict = self.lines
        for node_label in nodes_dict:
            node = nodes_dict[node_label]
            for connected_node in node.connected_nodes:
                line_label = node_label+connected_node
                line = lines_dict[line_label]
                line.successive[connected_node] = nodes_dict[connected_node]
                node.successive[line_label] = lines_dict[line_label]
        self._connected = True

    def propagate(self, signal_information, occupation= False):
        path = signal_information.path
        start_node = self.nodes[path[0]]
        propagated_signal_information = start_node.propagate(signal_information, occupation=False) #ricorsione
        return propagated_signal_information

    def set_weighted_paths(self, signal_power):
        if not self._connected:
            self.connect()
        node_labels = self.nodes.keys()
        pairs = []
        for label1 in node_labels:
            for label2 in node_labels:
                if label1 != label2:
                    pairs.append(label1 + label2)

        df = pd.DataFrame()
        paths = []
        latencies = []
        noises = []
        snrs = []

        for pair in pairs:
            for path in self.find_paths(pair[0], pair[1]):
                path_string = ''
                for node in path:
                    path_string = path_string + node + '->'
                paths.append(path_string[:-2])

                # propagation

                signal_information = Signalinformations(signal_power, path)
                signal_information = self.propagate(signal_information, occupation=False)
                latencies.append(signal_information.latency)
                noises.append(signal_information.noise_power)
                snrs.append(10 * np.log10(signal_information.signal_power / signal_information.noise_power))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df

    def available_path(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths(1)
        all_paths=[]
        for path in self.weighted_paths.path.values:
            if (path[0] == input_node) and (path[-1] == output_node):
                all_paths.append(path)
        unavailable_lines = [line for line in self.lines if self.lines[line].state == 'occupied']
        available_paths = []
        for path in all_paths:
            available = True
            for line in unavailable_lines:
                if (line[0] + '->' + line[1]) in path:
                    available = False
                    break
                if available:
                    available_paths.append(path)

        return available_paths

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_path(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0].replace('->', '')
        else:
            best_path = None

        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_path(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0].replace('->', '')
        else:
            best_path = None

        return best_path

    def stream(self, connections, best='latency'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            signal_power = connection.signal_power
            self.set_weighted_paths(signal_power)
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('error: best input not recognized. Value:', best)
                continue
            if path:
                in_signal_information = Signalinformations(signal_power, path)
                out_signal_information = self.propagate(in_signal_information, True)
                connection.latency = out_signal_information.latency
                noise = out_signal_information.noise_power
                connection.snr = 10 * np.log10(signal_power / noise)
            else:
                connection.latency = None
                connection.snr = 0

            streamed_connections.append(connection)
        return streamed_connections

class Connection(object):
    def __init__(self, input_node, output_node, signal_power):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = signal_power
        self._latency = 0
        self._snr = 0

    @property
    def input_node(self):
        return self._input_node

    @property
    def output_node(self):
        return self._output_node

    @property
    def signal_power(self):
        return self._signal_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr





#fine delle defnizioni dei classi e dei metodi
#main

from random import shuffle

network = Network('nodes.json')
network.connect()
node_labels = list (network.nodes.keys())
connections = []
for i in range (100):
    shuffle(node_labels)
    connection = Connection(node_labels[0], node_labels[-1], 1)
    connections.append(connection)

streamed_connections = network.stream(connections, best='snr')
snrs = [connection.snr for connection in streamed_connections]
print(snrs)
plt.hist(snrs, bins=10)
plt.title('SNR distribution')
plt.show()