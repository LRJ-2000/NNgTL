# -*- coding: utf-8 -*-
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
from utils.task import Task, get_random_formula
from utils.buchi_parse import Buchi
from utils.workspace import Workspace
import datetime
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
from unbiased_tree import UnbiasedTree
from heuristic_tree import HeuristicTree
from neural_tree import NeuralTree
from construct_tree import construction_neural_tree
from construct_tree import construction_heuristic_tree
from construct_tree import construction_unbiased_tree
from utils.visualization import path_plot_continuous
import matplotlib.pyplot as plt
from termcolor import colored
import pickle
import sys
from utils.model import LTL_Net, BA_Predict
import torch
from torch_geometric.loader import DataLoader
from utils.dataset import Task_Data, Task_Data_Hetero
from torchviz import make_dot
from matplotlib import colors
from utils.testing_data import Testing_data, save_testing_data, load_testing_data
import csv
from torch_geometric.data import Batch
import os
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=8000)

# random_seed = 42
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--LTL', type=str, default=None, help='LTL formula')
parser.add_argument('--generate_data', type=int, default=None, help='Generate testing data')
parser.add_argument('--data_id', type=int, default=None, help='Data id')
parser.add_argument('--model_scale', type=str, default='large', help='Network model scale')
parser.add_argument('--visualize_path', action='store_true', help='Visualize the path')
parser.add_argument('--use_pretrained_model', action='store_true', help='Use pretrained model')
parser.add_argument('--save_data', action='store_true', help='Save data')
parser.add_argument('--test_algorithm', action='store_true', help='Test algorithm')
parser.add_argument('--test_unbiased', action='store_true', help='whether test unbiased algorithm')

parser.add_argument('--n_max', type=int, default=4000, help='Maximum number of iterations')
parser.add_argument('--max_time', type=int, default=200, help='Maximum time allowed for tree construction')
parser.add_argument('--size_N', type=int, default=200, help='Size of the workspace')
parser.add_argument('--is_lite', action='store_true', help='Lite version, excluding extending and rewiring')
parser.add_argument('--weight', type=float, default=0.2, help='Weight parameter')
parser.add_argument('--p_closest', type=float, default=0.9, help='Probability of choosing node q_p_closest')
parser.add_argument('--y_rand', type=float, default=0.8, help='Probability used when deciding the target point')
parser.add_argument('--step_size', type=float, default=0.8, help='Step size used in function near')
parser.add_argument('--p_BA_predict', type=float, default=0.8, help='Probability of using BA prediction')

args = parser.parse_args()

n_max = args.n_max
size_N = args.size_N
max_time = args.max_time

para = dict()
para['is_lite'] = args.is_lite
para['weight'] = args.weight
para['p_closest'] = args.p_closest
para['y_rand'] = args.y_rand
para['step_size'] = args.step_size
para['p_BA_predict'] = args.p_BA_predict

# cmap = colors.ListedColormap(
# ['white', 'green', 'pink', 'blue', 'magenta', 'yellow', 'cyan', 'skyblue', 'brown', 'pink', 'black'])
cmap = colors.ListedColormap(
    ['white', 'green', 'pink', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'gray', 'black'])


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

model_scale_dict = {
    'tiny': 8,
    'small': 16,
    'medium': 32,
    'large': 64
}


def test_on_one_data(LTL=None, test_data_path='./data/testing_data', data_id=None, type=3, save_data=False, algorithm_list=[HeuristicTree, NeuralTree, UnbiasedTree], model_scale='large', visualize_path=False, use_pretrained_model=True):
    if type == 1:  # generate random LTL and workspace
        # workspace
        LTL = get_random_formula()
        workspace = Workspace(size_N, size_N)
        workspace.generate_random_map(7, 0.05, 3)
        # workspace.workspace_plot()
        task = Task(workspace, LTL)
        test_data = Testing_data(LTL, workspace, task)
    elif type == 2:  # load data from file
        # check if the file exists
        if not os.path.exists(test_data_path + 'test_data' + str(data_id) + '.pkl'):
            print('The file does not exist')
            return
        test_data = load_testing_data(test_data_path, 'test_data' + str(data_id))
        task = test_data.task
        workspace = test_data.workspace
        LTL = test_data.LTL
    elif type == 3:  # generate random workspace, using given LTL
        # raise error if LTL is not given
        if LTL is None:
            print('LTL is not given')
            return
        workspace = Workspace(size_N, size_N)
        workspace.generate_random_map(7, 0.05, 3)
        task = Task(workspace, LTL)
        test_data = Testing_data(LTL, workspace, task)


    buchi = Buchi(task)
    buchi.construct_buchi_graph()
    buchi.get_minimal_length()
    success = buchi.get_feasible_accepting_state()

    if not success:
        return
    graph = buchi.buchi_graph
    b_init = graph.graph['init'][0]
    for b_init in graph.graph['init']:
        for b_accept in graph.graph['accept']:
            if buchi.min_length[(b_init, b_accept)] < 2:
                print('init and accept are too close !')
                return

    if NeuralTree in algorithm_list:
        start_time = datetime.datetime.now()
        max_length = max(length for length in buchi.min_length.values() if length < np.inf)

        # Precompute nodes and edges
        nodes = list(graph.nodes())
        edges = list(graph.edges())
        num_nodes = len(nodes)
        num_edges = len(edges)

        # Map nodes to ids and collect features
        node_to_id = {node: idx for idx, node in enumerate(nodes)}
        id_to_node = nodes

        # Pre-create arrays for features and types
        node_features = np.zeros((num_nodes, 3), dtype=np.float32)
        node_types = np.zeros(num_nodes, dtype=np.int8)

        # Convert init and accept nodes to sets for faster lookup
        init_nodes = set(graph.graph['init'])
        accept_nodes = set(graph.graph['accept'])

        # Precompute minimal lengths to accepting states
        min_lengths = {}
        for node in nodes:
            lengths = [
                buchi.min_length[(node, acc)]
                for acc in accept_nodes
                if buchi.min_length.get((node, acc), np.inf) < np.inf and buchi.min_length.get((acc, acc), np.inf) < np.inf
            ]
            min_lengths[node] = min(lengths) if lengths else -1

        # Compute node features and types
        for idx, node in enumerate(nodes):
            if node in init_nodes:
                node_features[idx, 0] = 1
                node_types[idx] = 1
            elif node in accept_nodes:
                node_features[idx, 1] = 1
                node_types[idx] = 2

            if min_lengths[node] != -1:
                node_features[idx, 2] = min_lengths[node] / max_length
            else:
                node_features[idx, 2] = -1

        # Build edge indices and features
        edge_index = np.zeros((2, num_edges), dtype=np.int64)
        edge_attr = np.zeros((num_edges, 7), dtype=np.float32)

        for idx, edge in enumerate(edges):
            edge_index[0, idx] = node_to_id[edge[0]]
            edge_index[1, idx] = node_to_id[edge[1]]
            truth = graph.edges[edge].get('truth', {})
            for i in range(7):
                ap = f'l{i+1}_1'
                if ap in truth:
                    edge_attr[idx, i] = 1 if truth[ap] else -1
                else:
                    edge_attr[idx, i] = 0

        edge_type = np.zeros(num_edges, dtype=np.int8)

        # Convert data to tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        node_type = torch.tensor(node_types, dtype=torch.int8)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        edge_type = torch.tensor(edge_type, dtype=torch.int8)

        # Encode the workspace
        workspace_map = np.zeros((size_N, size_N, 8), dtype=np.float32)
        workspace_array = np.array(workspace.workspace)

        # Mark obstacles and useful labels
        workspace_map[..., 0][workspace_array == 10] = 1
        for label in task.useful:
            value = int(label[1:])
            workspace_map[..., value][workspace_array == value] = 1

        # Mark initial position
        discrete_init = workspace.continuous_to_discrete(task.init)
        workspace_map[discrete_init[0], discrete_init[1], 0] = -1

        # Convert to heterogeneous graph
        D = Task_Data_Hetero()
        D['node'].x = x
        D['node'].node_type = node_type
        D['edge'].x = edge_attr
        D['edge'].edge_type = edge_type
        D['source'].x = torch.full((1, 128), 0.5)

        # Build edge connections
        num_edges = edge_index.shape[1]
        node_to_edge = torch.stack([edge_index[0], torch.arange(num_edges)]).long()
        edge_to_node = torch.stack([torch.arange(num_edges), edge_index[1]]).long()

        D['node', 'connect', 'edge'].edge_index = node_to_edge
        D['edge', 'connect', 'node'].edge_index = edge_to_node

        # Connect nodes and edges to source
        num_nodes = D['node'].x.shape[0]
        edge_count = D['edge'].x.shape[0]
        node_to_source = torch.stack([torch.arange(num_nodes), torch.zeros(num_nodes, dtype=torch.long)]).long()
        edge_to_source = torch.stack([torch.arange(edge_count), torch.zeros(edge_count, dtype=torch.long)]).long()

        D['node', 'connect', 'source'].edge_index = node_to_source
        D['edge', 'connect', 'source'].edge_index = edge_to_source

        D.map = torch.tensor(workspace_map, dtype=torch.float)

        # Add self-loops
        for node_type in D.x_dict.keys():
            num_nodes = D[node_type].x.shape[0]
            self_loop_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).long()
            D[node_type, 'self_loop', node_type].edge_index = self_loop_index

        network_process_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f'Network pre-processing time: {network_process_time} seconds')

        if use_pretrained_model:
            print('loading pretrained model')
            model_path1 = 'model/pretrained_model/LTL_Net/LTL_Net_pretrained.pth'
            model_path2 = 'model/pretrained_model/BA_Predict/BA_Predict_pretrained.pth'
            model1 = LTL_Net(D.metadata())
            model1.load_state_dict(torch.load(model_path1))
            model1.eval()

            model2 = BA_Predict(D.metadata())
            model2.load_state_dict(torch.load(model_path2))
            model2.eval()

        else:
            print('loading the newest model')
            model_path1 = 'model/LTL_Net'
            model_path2 = 'model/BA_Predict'
            # read the newest model from the folder with prefix file name 'LTL_Net_'+model_scale
            files = os.listdir(model_path1)
            files = [f for f in files if re.match(f'LTL_Net_{model_scale}', f)]
            files.sort()
            model_path1 = os.path.join(model_path1, files[-1])
            # read the newest model from the folder with prefix file name 'BA_Predict_'+model_scale
            files = os.listdir(model_path2)
            files = [f for f in files if re.match(f'BA_Predict_{model_scale}', f)]
            files.sort()
            model_path2 = os.path.join(model_path2, files[-1])
            # Load the models
            n1 = model_scale_dict[model_scale]
            model1 = LTL_Net(D.metadata(), n1)
            model1.load_state_dict(torch.load(model_path1))
            model1.eval()

            model2 = BA_Predict(D.metadata())
            model2.load_state_dict(torch.load(model_path2))
            model2.eval()


        # Prepare the data batch
        batch = Batch.from_data_list([D])

        # Make predictions
        with torch.no_grad():
            pred1 = model2(batch.map, batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            pred2 = model1(batch.map, batch.x_dict, batch.edge_index_dict, batch.batch_dict)

        # Process predictions
        prob_nodes = torch.sigmoid(pred1).squeeze().numpy()
        # the probability of each node being a candidate node
        prob_nodes = prob_nodes[:, 1]
        prob_nodes_dict = {id_to_node[i]: p for i, p in enumerate(prob_nodes)}

        # Get probability map
        prob = torch.sigmoid(pred2).squeeze().numpy()


        # fig1 = plt.figure()
        # plt.xticks([])
        # plt.yticks([])
        # plt.axis('off')
        # fig1.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        # fig1.set_size_inches((8, 8), forward=False)
        # # plt.figure(figsize=(5, 5))
        # norm = colors.PowerNorm(gamma=0.45)
        # plt.imshow(prob.T, cmap='plasma', norm=norm)
        # # plt.savefig('prob.png', dpi=1000)
        # plt.savefig('prob.svg', format='svg')
        # plt.show()

    # # workspace.visualize_workspace(task.init, "workspace")
    # workspace.workspace_plot('workspace_continuous')


    def run_tree_construction(tree_class, construction_func, label, *args):
        start = datetime.datetime.now()
        init_state = (task.init, b_init)
        init_label = task.init_label
        tree = tree_class(workspace, buchi, init_state, init_label, 'prefix', *args)
        cost_path, num_of_iter, num_of_nodes, first_path_length, first_time = construction_func(tree, n_max, max_time)
        time_taken = (datetime.datetime.now() - start).total_seconds()
        if len(tree.goals):
            print(f"{label} tree takes {time_taken} seconds")
            final_path_length = cost_path[0][0]
            pre_path = cost_path[0][1]
            if visualize_path:
                path_plot_continuous(pre_path, workspace, label)
            return num_of_iter, num_of_nodes, final_path_length, time_taken
        else:
            print(f"Couldn't find the path within predetermined number of iterations for {label} tree")
            return None

    # Initialize results dictionary
    results_dict = {
        'LTL': LTL,
        'heuristic': None,
        'neural': None,
        'unbiased': None
    }

    # Run heuristic tree
    if HeuristicTree in algorithm_list:
        heuristic_results = run_tree_construction(HeuristicTree, construction_heuristic_tree, 'heuristic', para)
        if heuristic_results:
            num_of_iter1, num_of_nodes1, final_path_length1, time_1 = heuristic_results
            results_dict['heuristic'] = {
                'num_of_iter': num_of_iter1,
                'num_of_nodes': num_of_nodes1,
                'final_path_length': final_path_length1,
                'time': time_1
            }
            if num_of_iter1 > 10 and save_data:
                save_testing_data(test_data, test_data_path, 'test_data')
        else:
            return None

    # Run neural tree
    if NeuralTree in algorithm_list:
        neural_results = run_tree_construction(NeuralTree, construction_neural_tree, 'neural', prob, prob_nodes_dict, para)
        if neural_results:
            num_of_iter2, num_of_nodes2, final_path_length2, time_2 = neural_results
            results_dict['neural'] = {
                'num_of_iter': num_of_iter2,
                'num_of_nodes': num_of_nodes2,
                'final_path_length': final_path_length2,
                'time': time_2
            }
            if time_2 < time_1:
                print(f'neural tree is faster on task {data_id}')
            else:
                print(f'heuristic tree is faster on task {data_id}')
        else:
            return None

    # Run unbiased tree
    if UnbiasedTree in algorithm_list:
        unbiased_results = run_tree_construction(UnbiasedTree, construction_unbiased_tree, 'unbiased', para)
        if unbiased_results:
            num_of_iter3, num_of_nodes3, final_path_length3, time_3 = unbiased_results
            results_dict['unbiased'] = {
                'num_of_iter': num_of_iter3,
                'num_of_nodes': num_of_nodes3,
                'final_path_length': final_path_length3,
                'time': time_3
            }
        else:
            return None

    if type == 2:
        return results_dict

    return True

def generate_testing_data(num_of_data, test_data_path='./data/testing_data/'):

    # Ensure the test data directory exists
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    # Get existing test data files
    existing_files = [f for f in os.listdir(test_data_path) if f.startswith('test_data') and f.endswith('.pkl')]
    num_existing_data = len(existing_files)


    while num_existing_data < num_of_data:
        # Generate test data by calling test_on_one_data
        test_on_one_data(type=1, save_data=True, test_data_path=test_data_path, algorithm_list=[HeuristicTree])
        existing_files = [f for f in os.listdir(test_data_path) if f.startswith('test_data') and f.endswith('.pkl')]
        num_existing_data = len(existing_files)

def test_algorithm(test_data_path='./data/testing_data/', algorithm_list=[HeuristicTree, NeuralTree, UnbiasedTree], model_scale='large', use_pretrained_model=True):
    # Get existing test data files
    existing_files = [f for f in os.listdir(test_data_path) if f.startswith('test_data') and f.endswith('.pkl')]
    num_existing_data = len(existing_files)

    # Ensure there is at least one test data file
    if num_existing_data == 0:
        print('No testing data available')
        return

    # Initialize results
    results = []

    # Initialize metric values dictionary
    metric_values = {}
    algorithms = ['heuristic', 'neural', 'unbiased']
    metrics = ['num_of_iter', 'num_of_nodes', 'final_path_length', 'time']
    for alg in algorithms:
        for metric in metrics:
            metric_name = f'{alg}_{metric}'
            metric_values[metric_name] = []

    # Test each data file
    for data_id in range(num_existing_data):
        print(f'Testing on data id: {data_id}')
        result = test_on_one_data(test_data_path=test_data_path, type=2, data_id=data_id, algorithm_list=algorithm_list, model_scale=model_scale, use_pretrained_model=use_pretrained_model)
        if result:
            results.append(result)
            # Collect metric values
            for alg in algorithms:
                alg_result = result.get(alg)
                if alg_result:
                    for metric in metrics:
                        value = alg_result.get(metric)
                        metric_name = f'{alg}_{metric}'
                        if value is not None:
                            metric_values[metric_name].append(value)

    # Build fieldnames based on results_dict structure
    fieldnames = ['LTL']
    for alg in algorithms:
        for metric in metrics:
            fieldnames.append(f'{alg}_{metric}')

    # Save results to a csv file
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            row = {}
            row['LTL'] = result.get('LTL')
            for alg in algorithms:
                alg_result = result.get(alg)
                if alg_result:
                    for metric in metrics:
                        row[f'{alg}_{metric}'] = alg_result.get(metric)
                else:
                    for metric in metrics:
                        row[f'{alg}_{metric}'] = None
            writer.writerow(row)

    # Calculate mean and variance for each metric, excluding top and bottom 10% data
    for metric_name, values in metric_values.items():
        values = np.array(values)
        if len(values) > 0:
            sorted_values = np.sort(values)
            n = len(sorted_values)
            lower_index = int(n * 0.1)
            upper_index = int(n * 0.9)
            if upper_index > lower_index:
                trimmed_values = sorted_values[lower_index:upper_index]
                mean = np.mean(trimmed_values)
                variance = np.var(trimmed_values)
                print(f'{metric_name}: mean={mean}, var={variance}')
            else:
                print(f'{metric_name}: not enough data')
        else:
            print(f'{metric_name}: no valid data')
    


if __name__ == '__main__':
    algorithm_list = [HeuristicTree, NeuralTree, UnbiasedTree] if args.test_unbiased else [HeuristicTree, NeuralTree]
    if args.LTL:
        print('Testing on LTL:', args.LTL)
        test_on_one_data(args.LTL, type=3, save_data=args.save_data, algorithm_list=algorithm_list, model_scale=args.model_scale, visualize_path=args.visualize_path, use_pretrained_model=args.use_pretrained_model)
    elif args.generate_data:
        print('Generating testing data')
        generate_testing_data(args.generate_data)
    elif args.data_id:
        print('Testing on data id:', args.data_id)
        test_on_one_data(type=2, data_id=args.data_id, algorithm_list=algorithm_list, model_scale=args.model_scale, visualize_path=args.visualize_path, use_pretrained_model=args.use_pretrained_model)
    elif args.test_algorithm:
        print('Testing algorithm on all testing data')
        test_algorithm(algorithm_list=algorithm_list, model_scale=args.model_scale, use_pretrained_model=args.use_pretrained_model)

    # test_on_one_data('[]<> e1 && (NOT e1 U e2) && <> e3 ', 0, type=3, save_data= True,algorithm_list=[HeuristicTree, NeuralTree])

