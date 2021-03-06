#lab 10

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import Planck, pi
from scipy.special import erfcinv
from random import shuffle
import itertools as it
from mpl_toolkits.mplot3d import Axes3D
import copy


class Lightpath(object):
    def __init__(self, path: str, channel=0, rs=32e9, df=50e9, transceiver='shanon'):
        self._signal_power = None
        self._path = path
        self._noise_power = 0
        self._latency = 0
        self._channel = channel
        self._rs = rs
        self._df = df
        self._snr = None
        self._transceiver = transceiver
        self._optimized_powers = {}
        self._bitrate = None

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        self._transceiver = transceiver

    @property
    def optimized_powers(self):
        return self._optimized_powers

    @optimized_powers.setter
    def optimized_power(self, optimized_powers):
        self._optimized_powers = optimized_powers

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        self._snr = snr

    def update_snr(self, snr):
        if self.snr is None:
            self.snr = snr
        else:
            self.snr = 1 / (1 / self.snr + 1 / snr)

    @property
    def rs(self):
        return self._rs

    @property
    def df(self):
        return self._df

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

    @property
    def channel(self):
        return self._channel

    @latency.setter
    def latency(self, latency):
        self._latency = latency

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

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

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]
            lightpath.next()
            lightpath.signal_power = lightpath.optimized_powers[line_label]
            lightpath = line.propagate(lightpath, occupation)
        return lightpath

    def optimize(self, lightpath):
        path= lightpath.path
        if len(path) > 1:
            line_label = path[:2]
            line = self.successive[line_label]

            ase = line.ase_generation()
            eta = line.nli_generation(1, lightpath.rs, lightpath.df)
            p_opt = (ase / (2*eta)) ** (1/3)  # Calculate optimum signal power
            lightpath.optimized_powers.update({line_label: p_opt})

            lightpath.next()
            node = line.successive[lightpath.path[0]]
            lightpath = node.optimize(lightpath)

        return lightpath




class Line(object):
    def __init__(self, line_dict):
        self._label = line_dict['label']
        self._lenght = line_dict['lenght']
        self._Nch = line_dict['Nch']
        self._state = ['free'] * self._Nch
        self._successive = {}
        self._amplifiers = int(np.ceil(self._lenght / 80e3))
        self._span_lenght = self._lenght / self._amplifiers
        self._gain = 20
        self._noise_figure = 5


        # Physical parameters of the fiber

        self._alpha = 4.6e-5
        self._beta = 6.58e-27
        self._gamma = 1.27e-3

        # Set Gain to transparency
        self._gain = self.transparency()

    @property
    def Nch(self):
        return self._Nch

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def amplifiers(self):
        return self._amplifiers

    @property
    def span_lenght(self):
        return self._span_lenght

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self,noise_figure):
        self._noise_figure=noise_figure

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
        state = [s.lower().strip() for s in state]
        if set(state).issubset(set(['free', 'occupied'])):
            self._state = state
        else:
            print('ERROR: line state not recognized. Value:', set(state)-set(['free', 'occupied']))

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        self._successive = successive

    def latency_generation(self):
        latency = self.lenght / ((3*10 ^ 8) * (2 / 3))
        return latency

    def noise_generation(self, lightpath):
        noise = self.ase_generation() + \
              self.nli_generation(lightpath.signal_power,lightpath.rs,lightpath.df)
        return noise

    def propagate(self, lightpath, occupation=False):
        # update latency
        latency = self.latency_generation()
        lightpath.add_latency(latency)

        # Update noise
        noise = self.noise_generation(lightpath)
        lightpath.add_noise(noise)

        # Update SNR
        snr = lightpath.signal_power / noise
        lightpath.update_snr(snr)


        # Update line state

        if not occupation:
            channel = lightpath.channel
            new_state = self.state.copy()
            new_state[channel] = 'occupied'
            self.state = new_state

        node = self.successive[lightpath.path[0]]
        lightpath = node.propagate(lightpath, occupation)

        return lightpath

    def ase_generation(self):
        gain_lin = 10 ** (self._gain / 10)
        noise_figure_lin = 10 ** (self._noise_figure / 10)
        N = self._amplifiers
        f = 193.4e12
        h = Planck
        Bn = 12.5e9
        ase_noise = N * h * f * Bn * noise_figure_lin * (gain_lin -1)
        return ase_noise

    def nli_generation(self, signal_power, Rs, df):
        Pch = signal_power
        Bn = 12.5e9
        loss = np.exp(-self.alpha*self._span_lenght)
        gain_lin = 10 ** (self.gain / 10)
        N_spans = self.amplifiers
        eta = 16 / (27 * pi) * np.log(pi ** 2 * self.beta * Rs ** 2 * self.Nch ** (2 * Rs / df) / (2 * self.alpha)) * self.gamma ** 2 / (4 * self.alpha * self.beta * Rs ** 3)
        nli_noise = N_spans * (Pch ** 3 * loss * gain_lin * eta * Bn)
        return nli_noise

    def transparency(self):
        gain = 10 * np.log10(np.exp(self.alpha * self.span_lenght))
        return gain

    def upgrade(self,db):
        noise_figure = self.noise_figure-db
        return noise_figure






class Network(object):
    def __init__(self, json_path, nch=10, upgrade_line='BD',upgrade_dB=3):
        node_json = json.load(open(json_path, 'r'))
        self._nodes = {}
        self._lines = {}
        self._connected = False
        self._weighted_paths = None
        self._route_space = None
        self._Nch = nch
        self._upgrade_line = upgrade_line
        self._upgrade_dB = upgrade_dB


        for node_label in node_json:  # create node instance
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
                line_dict['Nch'] = self.Nch
                line = Line(line_dict)
                self._lines[line_label] = line
                if line_label == 'BD':
                    line.noise_figure = line.noise_figure - upgrade_dB
                    self._lines[line_label] = line

        # if specifiedm upgrade line



    @property
    def upgrade_line(self):
        return self._upgrade_line

    @property
    def Nch(self):
        return self._Nch

    @property
    def nodes(self):
        return self._nodes

    @property
    def lines(self):
        return self._lines

    @property
    def weighted_paths(self):
        return self._weighted_paths

    @property
    def route_space(self):
        return self._route_space

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

    def propagate(self, lightpath, occupation=False):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        propagated_lightpath = start_node.propagate(lightpath, occupation) # ricorsione
        return propagated_lightpath

    def set_weighted_paths(self):
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

                lightpath = Lightpath(path)
                lightpath = self.optimization(lightpath)
                lightpath = self.propagate(lightpath, occupation=False)

                latencies.append(lightpath.latency)
                noises.append(lightpath.noise_power)
                snrs.append(10 * np.log10(lightpath.snr))

        df['path'] = paths
        df['latency'] = latencies
        df['noise'] = noises
        df['snr'] = snrs
        self._weighted_paths = df

        route_space = pd.DataFrame()
        route_space['path'] = paths
        for i in range(self.Nch):
            route_space[str(i)] = ['free']*len(paths)
        self._route_space = route_space

    def available_path(self, input_node, output_node):
        if self.weighted_paths is None:
            self.set_weighted_paths()
        all_paths = []
        for path in self.weighted_paths.path.values:
            if (path[0] == input_node) and (path[-1] == output_node):
                all_paths.append(path)
        available_paths = []
        for path in all_paths:
            path_occupancy = self.route_space.loc[self.route_space.path == path].T.values[1:]
            if 'free' in path_occupancy:
                available_paths.append(path)
        return available_paths

    def find_best_snr(self, input_node, output_node):
        available_paths = self.available_path(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_snr = np.max(inout_df.snr.values)
            best_path = inout_df.loc[inout_df.snr == best_snr].path.values[0]
        else:
            best_path = None
        return best_path

    def find_best_latency(self, input_node, output_node):
        available_paths = self.available_path(input_node, output_node)
        if available_paths:
            inout_df = self.weighted_paths.loc[self.weighted_paths.path.isin(available_paths)]
            best_latency = np.min(inout_df.latency.values)
            best_path = inout_df.loc[inout_df.latency == best_latency].path.values[0]
        else:
            best_path = None
        return best_path

    def stream(self, connections, best='latency', transceiver='shannon', type='first'):
        streamed_connections = []
        for connection in connections:
            input_node = connection.input_node
            output_node = connection.output_node
            if best == 'latency':
                path = self.find_best_latency(input_node, output_node)
            elif best == 'snr':
                path = self.find_best_snr(input_node, output_node)
            else:
                print('error: best input not recognized. Value:', best)
                continue
            if path:
                path_occupancy = self.route_space.loc[self.route_space.path == path].T.values
                channel = [i for i in range(len(path_occupancy)) if path_occupancy[i] == 'free'][0]
                path = path.replace('->', '')
                in_lightpath = Lightpath(path, channel, transceiver=transceiver)
                in_lightpath = self.optimization(in_lightpath)
                out_lightpath = self.propagate(in_lightpath, True)

                self.calculate_bitrate(out_lightpath)


                if out_lightpath.bitrate == 0.0:
                    [self.update_route_space(lp.path, lp.channel, 'free') for lp in connection.lightpaths]
                    connection.block_connection()
                else:
                    connection.set_connection(out_lightpath)
                    self.update_route_space(out_lightpath.path, out_lightpath.channel, 'occupied')
                    if connection.residual_rate_request > 0:
                        self.stream([connection], best, transceiver, type='iter')
            else:
                [self.update_route_space(lp.path, lp.channel, 'free') for lp in connection.lightpaths]
                connection.block_connection()
            streamed_connections.append(connection)
        return streamed_connections

    @staticmethod
    def path_to_line_set(path):
        path = path.replace('->', '')
        return set([path[i] + path[i+1] for i in range(len(path)-1)])

    def update_route_space(self, path, channel, state):
        all_paths = [self.path_to_line_set(p) for p in self.route_space.path.values]
        states = self.route_space[str(channel)]
        lines = self.path_to_line_set(path)
        for i in range(len(all_paths)):
            line_set = all_paths[i]
            if lines.intersection(line_set):
                states[i] = state
        self.route_space[str(channel)] = states

    def optimization(self, lightpath):
        path = lightpath.path
        start_node = self.nodes[path[0]]
        optimized_lightpath = start_node.optimize(lightpath)
        optimized_lightpath.path = path

        return optimized_lightpath

    def calculate_bitrate(self, lightpath, bert=1e-3):
        snr = lightpath.snr
        Bn = 12.5e9
        Rs = lightpath.rs

        if lightpath.transceiver.lower() == 'fixed-rate' :
            snrt = 2 * erfcinv(2 * bert) * (Rs / Bn)
            rb = np.piecewise(snr, [snr < snrt, snr >= snrt],[0,100])
            lightpath.bitrate = rb

        elif lightpath.transceiver.lower() == 'flex-rate':
            snrt1 = 2* erfcinv(2 ** bert) ** 2 * (Rs / Bn)
            snrt2 = (14 / 3) * erfcinv(3/2 * bert) ** 2 * (Rs / Bn)
            snrt3 = 10 * erfcinv(8 / 3 * bert) ** 2 * (Rs/Bn)

            cond1 = (snr < snrt1)
            cond2 = (snr >= snrt1 and snr < snrt2)
            cond3 = (snr >= snrt2 and snr < snrt3)
            cond4 = (snr >= snrt3)

            rb = np.piecewise(snr, [cond1, cond2, cond3, cond4], [0, 100, 200, 400])
            lightpath.bitrate = rb
        elif lightpath.transceiver.lower() == 'shannon':
            rb = 2 * Rs * np.log2(1 + snr *(Rs / Bn)) * 1e-9
            lightpath.bitrate = rb

        return lightpath.bitrate








    #def optimization(self, lightpath):
        # sets the lightpath power to the optimal power calculated on the first line of the path
        #first_line = lightpath.path[0:2]
       # line = self.lines[first_line]

        #ase = line.ase_generation()
        #eta = line.nli_generation(1, lightpath.rs, lightpath.df)
        #lightpath.signal_power = (ase / (2*eta)) ** (1 / 3)  # calculate optimum power

        return lightpath


class Connection(object):
    def __init__(self, input_node, output_node, rate_request=0):
        self._input_node = input_node
        self._output_node = output_node
        self._signal_power = None
        self._latency = 0
        self._snr = []
        self._bitrate = None
        self._rate_request = float(rate_request)
        self._residual_rate_request = float(rate_request)
        self._lightpaths = []

    @property
    def clear_lightpaths(self):
        self._lightpaths = []

    @property
    def rate_request(self):
        return self._rate_request

    @property
    def residual_rate_request(self):
        return self._residual_rate_request

    @property
    def lightpaths(self):
        return self._lightpaths

    @lightpaths.setter
    def lightpaths(self, lightpath):
        self._lightpaths.append(lightpath)

    @property
    def bitrate(self):
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate):
        self._bitrate = bitrate

    def calculate_capacity(self):
        return self.bitrate

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
        self._snr.append(snr)

    @signal_power.setter
    def signal_power(self, signal_power):
        self._signal_power = signal_power

    def calculate_capacity(self):
        self.bitrate = sum([lightpath.bitrate for lightpath in self.lightpaths])
        return self.bitrate

    def set_connection(self, lightpath):
        self.signal_power = lightpath.signal_power
        self.latency = max(self.latency, lightpath.latency)
        self.snr = 10 * np.log10(lightpath.snr)
        self.lightpaths = lightpath
        self._residual_rate_request = self._residual_rate_request - lightpath.bitrate

        return self

    def block_connection(self):
        self.latency = None
        self.snr = 0
        self.bitrate = 0
        self.clear_lightpaths

        return self


#main
def create_traffic_matrix(nodes, rate, multiplier=1):
    s = pd.Series(data=[0.0] * len(nodes), index=nodes)
    df = pd.DataFrame(float(rate * multiplier), index=s.index, columns=s.index, dtype=s.dtype)
    np.fill_diagonal(df.values, s)

    return df


def plot3dbars(t):
    fig = plt.figure()
    ax = fig.gca( projection='3d')
    x_data, y_data = np.meshgrid(np.arange(t.shape[1]), np.arange(t.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = t.flatten()
    ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)
    plt.show()

NMC = 5
node_pairs_realizations = []
stream_conn_list = []
lines_state_list = []

for i in range(NMC):
    print('monte carlo realization #{:d}'.format(i+1))
    network = Network('nodes(1).json', nch=10, upgrade_line='BD', upgrade_dB=3)
    network.connect()
    node_labels = list(network.nodes.keys())
    T = create_traffic_matrix(node_labels, 400, multiplier=5)
    t = T.values
    print(T)
    connections = []
    node_pairs = list(filter(lambda x: x[0] != x[1], list(it.product(node_labels, node_labels))))
    shuffle(node_pairs)
    node_pairs_realizations.append(copy.deepcopy(node_pairs))

    for node_pair in node_pairs:
        connection = Connection(node_pair[0], node_pair[-1], float(T.loc[node_pair[0], node_pair[-1]]))
        connections.append(connection)

    streamed_connections = network.stream(connections, best='snr', transceiver='shannon')
    stream_conn_list.append(streamed_connections)
    lines_state_list.append(network.lines)

network.draw()

snr_conns = []
rbl_conns = []
rbc_conns = []

for streamed_conn in stream_conn_list :
    snrs = []
    rbl = []
    [snrs.extend(connection.snr) for connection in streamed_conn]
    for connection in streamed_conn:
        for lightpath in connection.lightpaths:
            rbl.append(lightpath.bitrate)
    rbc = [connection.calculate_capacity() for connection in streamed_conn]
    snr_conns.append(snrs)
    rbl_conns.append(rbl)
    rbc_conns.append(rbc)

avg_snr_list = []
avg_rbl_list = []
traffic_list = []
[traffic_list.append(np.sum(rbl_list)) for rbl_list in rbl_conns]
[avg_rbl_list.append(np.mean(rbl_list))for rbl_list in rbl_conns]
[avg_snr_list.append(np.mean(list(filter(lambda x: x != 0, snr_list)))) for snr_list in snr_conns]
#congestion
lines_labels = list(lines_state_list[0].keys())
congestions = {label: [] for label in lines_labels}

for line_state in lines_state_list:
    for line_label, line in line_state.items():
        cong = line.state.count('occupied') / len(line.state)
        congestions[line_label].append(cong)
avg_congestion = {label: 'BD' for label in lines_labels}
for line_label, cong in congestions.items():
    avg_congestion[line_label] = np.mean(cong)

plt.bar(range(len(avg_congestion)), list(avg_congestion.values()), align='center')
plt.xticks(range(len(avg_congestion)),list(avg_congestion.keys()))
plt.show()

print('\n')
print('Line to upgrade: {}'.format(max(avg_congestion, key=avg_congestion.get)))
print('avg total traffic {:.2f} Tbps'.format(np.mean(traffic_list) * 1e-3))
print('avg lightpath bitrate: {:.2f} Gbps'.format(np.mean(avg_rbl_list)))
print('avg lightpath snr: {:.2f} dB'.format(np.mean(avg_snr_list)))
