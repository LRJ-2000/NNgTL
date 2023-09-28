import random
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from shapely.geometry import Polygon, Point
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PolygonPatch
from scipy.spatial import ConvexHull

# Define constants
EMPTY_CELL = 0
OBSTACLE_CELL = 10
PATH_CELL = 8
INIT_CELL = 9
LABEL_CELLS = list(range(1, 8))

CMAP = colors.ListedColormap(
    ['white', 'green', 'pink', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'red', 'black']
)
bounds = [EMPTY_CELL, *LABEL_CELLS, OBSTACLE_CELL]


def generate_polygon_around_point(center_point, random_points_count=5, radius=10 / 200):
    """Generate a convex polygon around a given point."""

    # Generate random points within a circle around the center point
    angles = np.random.uniform(0, 2 * np.pi, random_points_count)
    distances = radius * np.sqrt(np.random.uniform(0, 1, random_points_count))
    random_points_x = center_point[0] + distances * np.cos(angles)
    random_points_y = center_point[1] + distances * np.sin(angles)

    # Ensure points are within boundaries
    random_points_x = np.clip(random_points_x, 0, 1)
    random_points_y = np.clip(random_points_y, 0, 1)

    random_points = np.column_stack([random_points_x, random_points_y])

    # Compute convex hull of the random points
    hull = ConvexHull(random_points)

    return Polygon([random_points[i] for i in hull.vertices])


def get_label(x, y, workspace):
    """
    Return the label string corresponding to workspace.workspace[x][y].

    Parameters:
    - x: int
        The x-coordinate in the discrete workspace.
    - y: int
        The y-coordinate in the discrete workspace.
    - workspace: Workspace object
        The workspace in which to check for the label.

    Returns:
    - str
        The label at the specified (x, y) location.
    """
    if workspace.workspace[x][y] == OBSTACLE_CELL:
        return 'o'
    if workspace.workspace[x][y] == EMPTY_CELL:
        return ''
    return 'l' + str(workspace.workspace[x][y])


def get_label_continuous(x, workspace):
    """
    Return the label in continuous space for point x.

    Parameters:
    - x: tuple of float
        The coordinates in the continuous workspace.
    - workspace: Workspace object
        The workspace in which to check for the label.

    Returns:
    - str
        The label at the specified x location.
    """
    point = Point(x)
    for label, region in workspace.regions.items():
        if point.within(region):
            return label
    for region in workspace.obs:
        if point.within(region):
            return 'o'
    return ''


class Workspace:
    """Define the workspace where robots reside."""

    def __init__(self, n, m):
        self.N = n
        self.M = m
        self.workspace = [[EMPTY_CELL] * m for _ in range(n)]
        self.continuous_workspace = (1.0, 1.0)
        self.label_loc = []
        self.regions = {}  # key: label, value: region
        self.obs = []
        self.CMAP = colors.ListedColormap(
            ['white', 'green', 'pink', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'red', 'black'])

    def discrete_to_continuous(self, x):
        """Convert discrete coordinates to continuous."""
        return ((x[0] + 0.5) / self.N, (x[1] + 0.5) / self.M)

    def continuous_to_discrete(self, x):
        """Convert continuous coordinates to discrete."""
        return (int(x[0] * self.N), int(x[1] * self.M))

    def sample_in_region(self, x):
        """Sample a point in the region corresponding to the discrete coordinate x."""
        x_cont = self.discrete_to_continuous(x)
        return (
            random.uniform(x_cont[0] - 0.5 / self.N, x_cont[0] + 0.5 / self.N),
            random.uniform(x_cont[1] - 0.5 / self.M, x_cont[1] + 0.5 / self.M)
        )

    def check_obs_around(self, x, y, r):
        """Check if there are obstacles around a given coordinate."""
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if 0 <= i < self.N and 0 <= j < self.M and self.workspace[i][j] != EMPTY_CELL:
                    return False
        return True

    def assign_discrete_label(self, x, y, r, label_region, label):
        """Assign labels to discrete workspace based on a continuous region."""
        for i in range(x - r, x + r + 1):
            for j in range(y - r, y + r + 1):
                if 0 <= i < self.N and 0 <= j < self.M:
                    point = self.discrete_to_continuous((i, j))
                    point = Point(point)
                    if point.within(label_region):
                        self.workspace[i][j] = label

    def generate_random_map(self, num_of_label=7, ratio_of_obstacle=0.2, type_of_obstacle=1):
        """Generate a random map with specified obstacle types."""
        self.workspace = [[EMPTY_CELL] * self.M for _ in range(self.N)]

        # Type 1: Random 3x3 obstacles
        if type_of_obstacle == 1:
            for i in range(1, self.N - 1):
                for j in range(1, self.M - 1):
                    if random.random() <= ratio_of_obstacle:
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                self.workspace[i + dx][j + dy] = OBSTACLE_CELL

                        center = self.discrete_to_continuous((i, j))
                        obs_now = Polygon([
                            (center[0] - 1.5 / self.N, center[1] - 1.5 / self.M),
                            (center[0] - 1.5 / self.N, center[1] + 1.5 / self.M),
                            (center[0] + 1.5 / self.N, center[1] + 1.5 / self.M),
                            (center[0] + 1.5 / self.N, center[1] - 1.5 / self.M)
                        ])
                        self.obs.append(obs_now)

        # Type 2: Single rectangle obstacle
        elif type_of_obstacle == 2:
            x = random.randint(0, self.N - 150)
            y = random.randint(0, self.M - 60)
            length = random.randint(60, 150)
            width = random.randint(20, 60)
            for i in range(length):
                for j in range(width):
                    self.workspace[x + i][y + j] = OBSTACLE_CELL

            top_left = self.discrete_to_continuous((x, y))
            obs_now = Polygon([
                (top_left[0] - 0.5 / self.N, top_left[1] - 0.5 / self.M),
                (top_left[0] + (length - 0.5) / self.N, top_left[1] - 0.5 / self.M),
                (top_left[0] + (length - 0.5) / self.N, top_left[1] + (width - 0.5) / self.M),
                (top_left[0] - 0.5 / self.N, top_left[1] + (width - 0.5) / self.M)
            ])
            self.obs.append(obs_now)

        # Type 3: Two rectangle obstacles (with potential overlap)
        elif type_of_obstacle == 3:
            x1 = random.randint(0, self.N - 80)
            y1 = random.randint(0, self.M - 40)
            length1 = random.randint(60, 80)
            width1 = random.randint(20, 40)
            for i in range(length1):
                for j in range(width1):
                    self.workspace[x1 + i][y1 + j] = OBSTACLE_CELL

            top_left1 = self.discrete_to_continuous((x1, y1))
            obs_now1 = Polygon([
                (top_left1[0] - 0.5 / self.N, top_left1[1] - 0.5 / self.M),
                (top_left1[0] + (length1 - 0.5) / self.N, top_left1[1] - 0.5 / self.M),
                (top_left1[0] + (length1 - 0.5) / self.N, top_left1[1] + (width1 - 0.5) / self.M),
                (top_left1[0] - 0.5 / self.N, top_left1[1] + (width1 - 0.5) / self.M)
            ])
            self.obs.append(obs_now1)

            x2 = random.randint(0, self.N - 40)
            y2 = random.randint(0, self.M - 80)
            length2 = random.randint(20, 40)
            width2 = random.randint(60, 80)
            for i in range(length2):
                for j in range(width2):
                    self.workspace[x2 + i][y2 + j] = OBSTACLE_CELL

            top_left2 = self.discrete_to_continuous((x2, y2))
            obs_now2 = Polygon([
                (top_left2[0] - 0.5 / self.N, top_left2[1] - 0.5 / self.M),
                (top_left2[0] + (length2 - 0.5) / self.N, top_left2[1] - 0.5 / self.M),
                (top_left2[0] + (length2 - 0.5) / self.N, top_left2[1] + (width2 - 0.5) / self.M),
                (top_left2[0] - 0.5 / self.N, top_left2[1] + (width2 - 0.5) / self.M)
            ])
            self.obs.append(obs_now2)

        # Generating Labels
        for _ in range(num_of_label):
            while True:
                x = random.randint(1, self.N - 2)
                y = random.randint(1, self.M - 2)
                if self.check_obs_around(x, y, 15):
                    self.label_loc.append((x, y))
                    center = self.discrete_to_continuous((x, y))
                    label_region = generate_polygon_around_point(center)
                    self.assign_discrete_label(x, y, 15, label_region, _ + 1)
                    self.regions['l{}'.format(_ + 1)] = label_region
                    break

    def visualize_workspace(self, save_image_name="", cmap=None):
        """
        Visualize the discrete workspace.

        Parameters:
            save_image_name (str): Name of the image file to save the visualization. Defaults to an empty string.
            cmap (ListedColormap, optional): Color map for visualization. Defaults to a predefined colormap.
        """
        if cmap is None:
            cmap = colors.ListedColormap(
                ['white', 'green', 'pink', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'red', 'black'])

        fig, ax = plt.subplots()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        a = np.array(self.workspace).T
        ax.imshow(a, cmap=cmap)
        fig.set_size_inches((8, 8), forward=False)
        # plt.savefig(save_image_name + '.svg', format='svg')
        plt.show()

    def workspace_plot(self, save_image_name=""):
        """
        Visualize the continuous workspace with labeled regions and obstacles.

        Parameters:
            save_image_name (str): Name of the image file to save the visualization.
        """
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xlim((0, self.continuous_workspace[0]))
        ax.set_ylim((0, self.continuous_workspace[1]))
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.gca().set_aspect('equal', adjustable='box')

        # Plot labeled regions
        for label, region in self.regions.items():
            x, y = zip(*region.exterior.coords[:-1])
            ax.fill(x, y, 'c')
            ax.text(np.mean(x), np.min(y) - 0.01, r'${}_{{{}}}$'.format(label[0], label[1:]), fontsize=40)

        # Plot obstacles
        for obs_region in self.obs:
            x, y = zip(*obs_region.exterior.coords[:-1])
            ax.fill(x, y, '0.75')
            ax.text(np.mean(x), np.mean(y), 'o', fontsize=40)

        fig.set_size_inches((8, 8), forward=False)
        # plt.savefig(save_image_name + '.svg', format='svg')
        plt.show()


if __name__ == "__main__":
    # Create a workspace of size 200x200
    ws = Workspace(200, 200)

    # Generate a random map with 7 labels, obstacle ratio of 0.2, and obstacle type 1
    ws.generate_random_map(num_of_label=7, ratio_of_obstacle=0.2, type_of_obstacle=3)

    # Visualize the workspace in discrete space
    cmap = colors.ListedColormap(['white', 'green', 'pink', 'blue', 'gray', 'gray', 'gray', 'gray', 'gray', 'red', 'black'])
    ws.visualize_workspace("discrete_workspace", CMAP)

    # Visualize the workspace in continuous space
    ws.workspace_plot("continuous_workspace")

    # Get the label of a specific point in discrete space
    label = get_label(100, 100, ws)
    print(f"Label at (100,100) in discrete space: {label}")

    # Get the label of a specific point in continuous space
    label_cont = get_label_continuous((0.5, 0.5), ws)
    print(f"Label at (0.5,0.5) in continuous space: {label_cont}")
