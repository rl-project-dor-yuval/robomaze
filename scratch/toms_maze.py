import matplotlib
import csv
matplotlib.use('agg')

from matplotlib import pyplot

print(f'using pyplot backend: {pyplot.get_backend()}')
import numpy as np
from descartes import PolygonPatch
from shapely import speedups
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from PIL import Image
import os


class Maze:
    def __init__(self, grid_size, free_tiles, occupied_tiles, noise=None, points=None):
        self.use_speedups()
        # we partition the space to a grid, in the following members the points describe the indices and not the
        # coordinate values so point (0,4) is the 0th point according to x and 4th according to y
        self.grid_size = grid_size
        self.free_tiles = free_tiles
        self.occupied_tiles = occupied_tiles

        # we also keep the points themselves
        self.points = self._get_points(grid_size, noise, points)

        # some private members to allow fast computations
        self._polygons = {}
        self._occupied_union = None

    @staticmethod
    def use_speedups():
        if speedups.available:
            speedups.enable()

    def _get_points(self, grid_size, noise, points):
        if points is not None:
            assert noise is None, 'if "points" were provided, noise should not be provided'
            return points

        if grid_size == 0:
            return {}

        # if points=None, we need to generate the list of points.
        def _get_loc(c):
            if noise is not None:
                # if noise is not none, we add uniform noise to each point
                c += np.random.uniform(-noise, noise)
            c = min(max(c, 0), grid_size)
            result = 2. * c / self.grid_size - 1.
            return result

        points = {
            (x, y): (_get_loc(x), _get_loc(y))
            for x in range(grid_size + 1) for y in range(grid_size + 1)
        }

        return points

    def get_data(self):
        return self.grid_size, self.free_tiles, self.occupied_tiles, self.points

    def sample_free_state(self):
        free_polygon_areas = [self._get_polygon(x, y).area for (x, y) in self.free_tiles]
        total_areas = sum(free_polygon_areas)
        p = [area / total_areas for area in free_polygon_areas]
        free_tile_index = np.random.choice(range(len(self.free_tiles)), p=p)
        # partition into triangles
        x, y = self.free_tiles[free_tile_index]
        points1 = [self.points[(x, y)], self.points[(x, y + 1)], self.points[(x + 1, y + 1)]]
        points2 = [self.points[(x, y)], self.points[(x + 1, y)], self.points[(x + 1, y + 1)]]
        triangle1 = Polygon(points1, [])
        triangle2 = Polygon(points2, [])
        area1 = triangle1.area
        area2 = triangle2.area
        is_first = np.random.uniform() < (area1 / (area1 + area2))
        points = points1 if is_first else points2
        triangle = triangle1 if is_first else triangle2
        p1, p2, p3 = [np.array(p) for p in points]
        p2_ = p2 - p1
        p3_ = p3 - p1
        while True:
            # according to https://mathworld.wolfram.com/TrianglePointPicking.html
            sample = np.random.uniform() * p2_ + np.random.uniform() * p3_
            sample = sample + p1
            if triangle.contains(Point(sample)):
                assert not self.is_collision(sample)
                return sample

    def _get_occupied_union(self):
        if self._occupied_union is None:
            polygons = [self._get_polygon(x, y) for (x, y) in self.occupied_tiles]
            self._occupied_union = unary_union(polygons)
        return self._occupied_union

    def is_collision(self, state1, state2=None):
        if state2 is None:
            intersection_object = Point(state1)
        else:
            intersection_object = LineString([state1, state2])
        return self._get_occupied_union().intersects(intersection_object)

    def _get_polygon(self, x, y) -> Polygon:
        key = (x, y)
        if not key in self._polygons:
            self._polygons[key] = Polygon(
                [
                    self.points[(x, y)], self.points[(x, y + 1)],
                    self.points[(x + 1, y + 1)], self.points[(x + 1, y)]
                ], [])
        return self._polygons[key]

    @staticmethod
    def _plot_path(path, path_color, ax):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        # ax.plot(xs, ys, '.-', color=path_color, alpha=0.3)
        ax.plot(xs, ys, '-', color=path_color, alpha=0.7)

    @staticmethod
    def _plot_point(point, path_color, ax):
        ax.scatter(point[0], point[1], marker='X', color=path_color, alpha=0.7)

    def plot(self, paths=None, points=None, save_path=None):
        fig = self._create_figure(paths=paths, points=points)
        data = Maze._figure_to_data(fig)
        if save_path is not None:
            img = Image.fromarray(data, 'RGB')
            img.save(save_path)
        return fig, data

    def plot_obstacles(self, ax):
        for (x, y) in self.occupied_tiles:
            patch = PolygonPatch(
                self._get_polygon(x, y), facecolor='#6699cc', edgecolor='#6699cc',
                alpha=1.0, zorder=2
            )
            ax.add_patch(patch)

    def _create_figure(self, include_obstacles=True, paths=None, points=None, display=False):
        pyplot.clf()
        fig = pyplot.figure(dpi=90)
        ax = fig.add_subplot(111)

        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'tan', 'teal', 'pink']
        if paths is not None:
            for i, p in enumerate(paths):
                color = colors[i % len(colors)]
                self._plot_path(p, color, ax)

        if points is not None:
            for i, p in enumerate(points):
                color = colors[i % len(colors)]
                self._plot_point(p, color, ax)

        if include_obstacles:
            self.plot_obstacles(ax)

        ax.set_xlim(-1., 1.)
        ax.set_ylim(-1., 1.)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        fig.tight_layout(pad=0, h_pad=0, w_pad=0)
        pyplot.subplots_adjust(left=0.0, right=1., top=1., bottom=0.0, hspace=0, wspace=0)
        pyplot.draw()

        if display:
            pyplot.show()

        return fig

    @staticmethod
    def _figure_to_data(fig):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        dims = fig.canvas.get_width_height()[::-1]
        data = data.reshape(dims + (3,))
        crop = dims[1] - dims[0]
        data = data[:, int(crop / 2):-int(crop / 2), :]
        return data

    def create_visual_context(self, size=None, display=False):
        fig = self._create_figure(include_obstacles=True, paths=None)
        if display:
            pyplot.show()
        data = Maze._figure_to_data(fig)
        pyplot.close('all')
        # img = Image.fromarray(data, 'RGB')
        # img_grey = img.convert('L')
        # img_grey = img.convert('1')
        binary_data = data.sum(axis=-1) == 255 * 3
        img_grey = Image.fromarray(binary_data)
        if size is not None:
            img_grey = img_grey.resize((size, size))
        result = np.array(img_grey).astype(np.uint8)
        result *= 255
        if result.min() == 255:
            print(f'data min {data.min()}')
            print(f'grey image min {np.array(img_grey).min()}')
            assert False, 'error: image is all black'
        return result

    def create_state_map(self, state, size=None):
        fig = self._create_figure(include_obstacles=False, points=[state])
        data = Maze._figure_to_data(fig)
        pyplot.close('all')
        img = Image.fromarray(data, 'RGB')
        img_grey = img.convert('L')
        if size is not None:
            img_grey = img_grey.resize((size, size))
        return np.array(img_grey)


def generate_random_maze(grid_size=20, open_area_ratio=0.5, contagion_factor=0.7, noise=0.2) -> Maze:
    all_tiles = grid_size * grid_size
    number_of_open_tiles = int(np.ceil(all_tiles * open_area_ratio))

    occupied_tiles = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    free_tiles = []

    def get_random_next(current):
        next = list(current)[:]
        action = np.random.randint(0, 4)
        if action == 0:
            next[0] += 1
        elif action == 1:
            next[0] -= 1
        elif action == 2:
            next[1] += 1
        elif action == 3:
            next[1] -= 1
        else:
            assert False, 'invalid action'
        return tuple(next)

    def is_in(item, list_of_items):
        try:
            item_index = list_of_items.index(item)
            return True, item_index
        except:
            return False, None

    # pop an occupied tile
    current_index = np.random.randint(0, len(occupied_tiles))
    current = occupied_tiles.pop(current_index)
    # mark as free
    free_tiles.append(current)
    while len(free_tiles) < number_of_open_tiles:
        free_tile_index = np.random.randint(0, len(free_tiles))
        current = free_tiles[free_tile_index]
        # start a random dfs from it
        next = get_random_next(current)
        # while next is legal
        while 0 <= next[0] < grid_size and 0 <= next[1] < grid_size:
            if len(free_tiles) == number_of_open_tiles:
                # if we have enough tiles break
                break
            if np.random.uniform() > contagion_factor:
                # see if we want the chain to continue
                break
            if is_in(next, free_tiles)[0]:
                # see if next is already free
                break
            free_tiles.append(next)
            in_occupied, occupied_index = is_in(next, occupied_tiles)
            assert in_occupied
            occupied_tiles.pop(occupied_index)
            next = get_random_next(next)

    return Maze(grid_size, free_tiles, occupied_tiles, noise)


def load_maze_from_data(data_from_get_data) -> Maze:
    maze = Maze(grid_size=0, free_tiles=[], occupied_tiles=[], noise=None, points=None)
    maze.grid_size, maze.free_tiles, maze.occupied_tiles, maze.points = data_from_get_data
    return maze


if __name__ == '__main__':
    mz = generate_random_maze(8, 0.5, 0.99, 0.25)
    s = mz.sample_free_state()
    g = mz.sample_free_state()
    paths = [
        [(0., 0.), (0.5, 0.5)],
        [s, g],
        [(0, 0), (5, 0.25)]
    ]
    _, data = mz.plot(paths=paths, save_path=".\\data.png")
    print(data.shape)

    data1 = mz.create_visual_context()
    data2 = mz.create_state_map(s)
    data3 = mz.create_state_map(g)
    Image.fromarray(data1, 'L').save('.\\data1.png')
    Image.fromarray(data2, 'L').save('.\\data2.png')
    Image.fromarray(data3, 'L').save('.\\data3.png')

    data4 = mz.create_visual_context(size=200)
    data5 = mz.create_state_map(s, size=200)
    data6 = mz.create_state_map(g, size=200)
    Image.fromarray(data4, 'L').save('.\\data4.png')
    Image.fromarray(data5, 'L').save('.\\data5.png')
    Image.fromarray(data6, 'L').save('.\\data6.png')

    # Saving the matrices with the start & goal locations
    # with open(os.path.join(os.path.curdir, "start_loc_data.csv"), 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data5)
    # with open(os.path.join(os.path.curdir, "goal_loc_data.csv"), 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(data6)

    print('here')
