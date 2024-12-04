# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from .workspace import get_label
from termcolor import colored
from matplotlib import colors
from matplotlib.patches import Polygon as PolygonPatch

EMPTY_CELL = 0
OBSTACLE_CELL = 10
LABEL_CELL_1 = 1
LABEL_CELL_2 = 2
LABEL_CELL_3 = 3
LABEL_CELL_4 = 4
LABEL_CELL_5 = 5
LABEL_CELL_6 = 6
LABEL_CELL_7 = 7
PRE_PATH_CELL = 8
SUF_PATH_CELL = 9

cmap = colors.ListedColormap(['white', 'green', 'red', 'blue', 'skyblue', 'brown', 'yellow', 'black', 'purple', 'gray'])
bounds = [EMPTY_CELL, LABEL_CELL_1, LABEL_CELL_2, LABEL_CELL_3, LABEL_CELL_4, LABEL_CELL_5, LABEL_CELL_6, LABEL_CELL_7, PRE_PATH_CELL, SUF_PATH_CELL, OBSTACLE_CELL]
norm = colors.BoundaryNorm(bounds, cmap.N)

def path_plot_continuous(path, workspace, path_type='prefix'):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim((0, workspace.continuous_workspace[0]))
    ax.set_ylim((0, workspace.continuous_workspace[1]))
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(b=True, which='major', color='k', linestyle='--')
    for key in workspace.regions.keys():
        c = 'c'
        x = []
        y = []
        patches = []
        for point in list(workspace.regions[key].exterior.coords)[:-1]:
            x.append(point[0])
            y.append(point[1])
        polygon = PolygonPatch(np.column_stack((x, y)), True)
        patches.append(polygon)
        p = PatchCollection(patches, facecolors=c, edgecolors=c)
        ax.add_collection(p)
        ax.text(np.mean(x), np.min(y)-0.01, r'${}_{{{}}}$'.format(key[0], key[1:]), fontsize=40)
    for key in workspace.obs:
        c = '0.75'
        x = []
        y = []
        patches = []
        for point in list(key.exterior.coords)[:-1]:
            x.append(point[0])
            y.append(point[1])
        polygon = PolygonPatch(np.column_stack((x, y)), True)
        patches.append(polygon)
        p = PatchCollection(patches, facecolors=c, edgecolors=c)
        ax.add_collection(p)
        ax.text(np.mean(x), np.mean(y), 'o', fontsize=40)
    fig.set_size_inches((8, 8), forward=False)

    c = 'red' if path_type == 'prefix' else 'blue'
    path_x = np.asarray([point[0][0] for point in path])
    path_y = np.asarray([point[0][1] for point in path])
    ax.plot(path_x, path_y, color=c, linewidth=2)
    plt.savefig('path.svg', format='svg')
    plt.show()

def path_print(path, workspace):
    """
    print the path
    :param path: found path
    :param workspace: workspace
    :return: printed path of traversed regions. points with empty label are depicted as dots
    """
    print('robot 1: ', end='')
    # prefix path, a path of x's or y's of a robot
    x_pre = [point[0][0] for point in path[0]]
    y_pre = [point[0][1] for point in path[0]]
    path_print_helper(x_pre, y_pre, workspace)
    # suffix path
    x_suf = [point[0][0] for point in path[1]]
    y_suf = [point[0][1] for point in path[1]]
    path_print_helper(x_suf, y_suf, workspace)
    print('')


def path_print_helper(x, y, workspace):
    """
    help to print the path
    :param x: a path of x's of a robot throughout the run
    :param y: a path of y's of a robot throughout the run
    :param workspace: workspace
    :return: printed path of traversed regions. points with empty label are depicted as dots
    """
    for i in range(len(x)):
        label = get_label(x[i], y[i], workspace)
        label = ' .' if not label else label
        print(label + ' --> ', end='')
    print(colored('|| ', 'yellow'), end='')
