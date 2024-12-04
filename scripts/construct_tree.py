import numpy as np
import pyvisgraph as vg
import datetime


def construction_heuristic_tree(tree, n_max):
    """
    construction of the heuristic tree
    :param tree: heuristic-tree
    :param n_max: maximum number of iterations
    :return: found path
    """
    # trivial suffix path, the initial state can transition to itself
    if tree.segment == 'suffix' and tree.check_transition_b(tree.init[1], tree.heuristic_tree.nodes[tree.init]['label'],
                                                            tree.init[1]):
        tree.goals.add(tree.init)
        if tree.buchi.buchi_graph.edges[(tree.init[1], tree.init[1])]['truth'] == '1':
            return {0: [0, []]}
        else:
            return {0: [0, [tree.init]]}
    s = datetime.datetime.now()
    flag = False
    num_of_iter = 0
    first_path_length = 0
    first_time = 0
    for n in range(n_max):
        if (datetime.datetime.now() - s).total_seconds() > 2000:  # or n > 2000:
            print('overtime')
            return 0, n, tree.heuristic_tree.number_of_nodes(), 100000, (datetime.datetime.now() - s).total_seconds()

        x_new, q_p_closest = tree.biased_sample()
        if not x_new: continue

        ap = tree.get_label(x_new)
        ap = ap + '_' + str(1) if ap != '' else ap
        label = ap

        # near state
        if tree.lite:
            # avoid near
            near_nodes = [q_p_closest]
        else:
            near_nodes = tree.near(x_new)
            near_nodes = near_nodes + [q_p_closest] if q_p_closest not in near_nodes else near_nodes
        # check the line is obstacle-free
        obs_check = tree.obstacle_check(near_nodes, x_new, label)

        for b_state in tree.buchi.buchi_graph.nodes:
            # new product state
            q_new = (x_new, b_state)
            # extend
            added = tree.extend(q_new, near_nodes, label, obs_check)
            # rewire
            if not tree.lite and added:
                tree.rewire(q_new, near_nodes, obs_check)

        # detect the first accepting state
        if len(tree.goals) > 0 and tree.segment == 'prefix' and flag == False:
            flag = True
            paths = tree.find_path(tree.goals)
            print("heuristic RRT finds first goal after {} iterations:".format(n))
            num_of_iter = n
            print("the path length is {}".format(paths[0][0]))
            first_path_length = paths[0][0]
            first_time = (datetime.datetime.now() - s).total_seconds()
            break
        if len(tree.goals) > 0 and tree.segment == 'suffix':
            print('find first path after {} iterations'.format(n), end=' ')
            break
    return tree.find_path(tree.goals), num_of_iter, tree.heuristic_tree.number_of_nodes(), first_path_length, first_time


def path_via_visibility(tree, path):
    """
    using the visibility graph to find the shortest path
    :param tree: heuristic tree
    :param path: path found by the first step of the suffix part
    :return: a path in the free workspace (after treating all regions as obstacles) and its distance cost
    """
    # find a path using visibility graph method
    init = path[-1][0]
    goal = path[0][0]
    shortest = tree.g.shortest_path(vg.Point(init[0], init[1]), vg.Point(goal[0], goal[1]))
    path_free = [(shortest[i], '') for i in range(len(shortest))]  # second component serves as buchi state
    cost = 0
    for i in range(1, len(shortest)):
        cost = cost + np.linalg.norm(np.subtract(shortest[i], shortest[i - 1]))
    return cost, path_free
