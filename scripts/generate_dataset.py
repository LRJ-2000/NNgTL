import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import torch
import numpy as np
from utils.task import Task, get_random_formula
from utils.buchi_parse import Buchi
from utils.workspace import Workspace
from heuristic_tree import HeuristicTree
from construct_tree import construction_heuristic_tree
from utils.dataset import Task_Data, Task_Data_Hetero
from utils.testing_data import Testing_data
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--save_dir', type=str, default='data/training_data')


parser.add_argument('--n_max', type=int, default=4000, help='Maximum number of iterations')
parser.add_argument('--max_time', type=int, default=200, help='Maximum time allowed for tree construction')
parser.add_argument('--size_N', type=int, default=200, help='Size of the workspace')
parser.add_argument('--is_lite', action='store_true', help='Lite version, excluding extending and rewiring')
parser.add_argument('--weight', type=float, default=0.2, help='Weight parameter')
parser.add_argument('--p_closest', type=float, default=0.9, help='Probability of choosing node q_p_closest')
parser.add_argument('--y_rand', type=float, default=0.8, help='Probability used when deciding the target point')
parser.add_argument('--step_size', type=float, default=0.8, help='Step size used in function near')

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

def GenericBresenhamLine(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    s1 = 1 if x2 > x1 else -1
    s2 = 1 if y2 > y1 else -1
    swapped = dy > dx
    if swapped:
        dx, dy = dy, dx

    e = 2 * dy - dx
    x, y = x1, y1
    points = [(x, y)]

    for _ in range(dx):
        if e >= 0:
            if swapped:
                x += s1
            else:
                y += s2
            e -= 2 * dx
        if swapped:
            y += s2
        else:
            x += s1
        e += 2 * dy
        points.append((x, y))

    return points

def generate_training_data(num_samples, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    existing_files = [f for f in os.listdir(save_dir) if f.startswith('training_data_') and f.endswith('.pkl')]
    start_index = len(existing_files)
    
    while start_index < num_samples:
        # Generate random LTL and workspace
        LTL = get_random_formula()
        workspace = Workspace(size_N, size_N)
        workspace.generate_random_map()
        task = Task(workspace, LTL)
        test_data = Testing_data(LTL, workspace, task)

        buchi = Buchi(task)
        buchi.construct_buchi_graph()
        buchi.get_minimal_length()
        success = buchi.get_feasible_accepting_state()
        if not success:
            continue
        graph = buchi.buchi_graph

        b_init = graph.graph['init'][0]
        init_state = (task.init, b_init)
        init_label = task.init_label

        b_accept = graph.graph['accept'][0]
        if buchi.min_length[(b_init, b_accept)] < 2:
            print('init and accept are too close !')
            continue

        # Use heuristic tree to generate labels
        tree_pre = HeuristicTree(workspace, buchi, init_state, init_label, 'prefix', para)
        cost_path_pre, num_of_iter, num_of_nodes, first_path_length, first_time = construction_heuristic_tree(tree_pre, n_max)

        if len(tree_pre.goals):
            if first_path_length < 0.5:
                print('too short path')
                continue

            # Prepare data for training
            max_length = max(length for length in buchi.min_length.values() if length < np.inf)

            # Map nodes to ids and collect features
            node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
            id_to_node = list(graph.nodes())
            node_features = []
            node_types = []

            for node in graph.nodes():
                feature = np.zeros(3)
                if node in graph.graph['init']:
                    feature[0] = 1
                    node_type = 1
                elif node in graph.graph['accept']:
                    feature[1] = 1
                    node_type = 2
                else:
                    node_type = 0

                # Compute minimal length to accepting states
                min_lengths = [
                    buchi.min_length[(node, acc)]
                    for acc in graph.graph['accept']
                    if buchi.min_length[(node, acc)] < np.inf and buchi.min_length[(acc, acc)] < np.inf
                ]
                if min_lengths:
                    feature[2] = min(min_lengths) / max_length
                else:
                    feature[2] = -1

                node_features.append(feature)
                node_types.append(node_type)

            # Build edge indices and features
            edge_index_in = [node_to_id[edge[0]] for edge in graph.edges()]
            edge_index_out = [node_to_id[edge[1]] for edge in graph.edges()]
            edge_index = [edge_index_in, edge_index_out]

            edge_attr = []
            for edge in graph.edges():
                feature = np.zeros(7)
                truth = graph.edges[edge].get('truth', {})
                for i in range(7):
                    ap = f'l{i+1}_1'
                    if ap not in truth:
                        feature[i] = 0
                    elif truth[ap]:
                        feature[i] = 1
                    else:
                        feature[i] = -1
                edge_attr.append(feature)

            edge_type = np.zeros(len(edge_index_in))

            # Convert data to tensors
            node_features = np.array(node_features)
            node_types = np.array(node_types)
            edge_index = np.array(edge_index)
            edge_attr = np.array(edge_attr)
            edge_type = np.array(edge_type)
            x = torch.tensor(node_features, dtype=torch.float)
            node_type = torch.tensor(node_types, dtype=torch.int8)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            edge_type = torch.tensor(edge_type, dtype=torch.int8)

            # Encode the workspace
            workspace_map = np.zeros((size_N, size_N, 8))
            workspace_array = np.array(workspace.workspace)

            # Mark obstacles and useful labels
            workspace_map[..., 0][workspace_array == 10] = 1
            for label in task.useful:
                value = int(label[1:])  # Extract numeric part of 'lX'
                mask = (workspace_array == value)
                workspace_map[mask, value] = 1

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

            pre_path = cost_path_pre[0][1]
            point_list = []
            state_list = []
            label_path = np.zeros((size_N, size_N))
            for i in range(1, len(pre_path)):
                discrete_point_1 = workspace.continuous_to_discrete(pre_path[i-1][0])
                discrete_point_2 = workspace.continuous_to_discrete(pre_path[i][0])
                point_list.extend(GenericBresenhamLine(discrete_point_1[0], discrete_point_1[1], discrete_point_2[0], discrete_point_2[1]))
                state_list.append(node_to_id[pre_path[i - 1][1]])
            state_list.append(node_to_id[pre_path[len(pre_path) - 1][1]])

            for x in point_list:
                if workspace_array[x[0], x[1]] != 10:
                    label_path[x[0]][x[1]] = 1
                if x[0] >= 1 and workspace_array[x[0] - 1, x[1]] != 10:
                    label_path[x[0] - 1][x[1]] = 1
                if x[0] < size_N - 1 and workspace_array[x[0] + 1, x[1]] != 10:
                    label_path[x[0] + 1][x[1]] = 1
                if x[1] >= 1 and workspace_array[x[0], x[1] - 1] != 10:
                    label_path[x[0]][x[1] - 1] = 1
                if x[1] < size_N - 1 and workspace_array[x[0], x[1] + 1] != 10:
                    label_path[x[0]][x[1] + 1] = 1
            D.label = torch.tensor(label_path, dtype=torch.float)

            # # visualize the label_path and discrete workspace
            # fig, ax = plt.subplots()
            # ax.imshow(workspace_array)
            # ax.imshow(label_path, alpha=0.5)
            # plt.show()


            node_label = []
            for node in graph.nodes:
                if node_to_id[node] in state_list:
                    node_label.append([0, 1])
                else:
                    node_label.append([1, 0])
            D.node_label = torch.tensor(node_label, dtype=torch.float)

            # Save each sample separately
            sample_path = os.path.join(save_dir, f'training_data_{start_index}.pkl')
            with open(sample_path, 'wb') as f:
                pickle.dump(D, f)

            print(f"Sample {start_index} generated and saved to {sample_path}")
            start_index += 1

    print(f"All training data generated and saved to {save_dir}")

if __name__ == '__main__':
    num_samples = args.num_samples
    save_dir = args.save_dir
    generate_training_data(num_samples, save_dir)