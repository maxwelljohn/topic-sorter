import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict


class TSPProblem(object):
    def __init__(self, filepath):
        assert filepath.endswith('.tsp')
        with open(filepath, 'r') as infile:
            self.info = defaultdict(list)
            line = infile.readline()
            while not line.startswith('NODE_COORD_SECTION'):
                parts = line.split(':')
                self.info[parts[0].strip()].append(':'.join(parts[1:]).strip())
                line = infile.readline()
            assert self.info['TYPE'] == ['TSP']
            assert self.info['EDGE_WEIGHT_TYPE'] == ['EUC_2D']
            assert len(self.info['DIMENSION']) == 1
            self.dimension = int(self.info['DIMENSION'][0])

            self.nodes = []
            for i in range(1, self.dimension+1):
                line = infile.readline().rstrip()
                # Unfortunately the .tsp files don't distinguish between
                # north vs south latitude or east vs west longitude.
                # So the TSP may be flipped in some cases.
                node_num, latitude, longitude = line.split(' ')
                assert int(node_num) == i
                self.nodes.append(
                    (float(longitude)/1000, float(latitude)/1000)
                )
            line = infile.readline().rstrip()
            assert line == '' or line == 'EOF'

    def show(self):
        xs = [node[0] for node in self.nodes]
        ys = [node[1] for node in self.nodes]

        # Ensure x & y use same scale, for accurate distance visualization.
        x_spread = max(xs)-min(xs)
        x_mid = (max(xs)+min(xs))/2
        y_spread = max(ys)-min(ys)
        y_mid = (max(ys)+min(ys))/2
        buff = (max(x_spread, y_spread) * 1.1)/2
        bounds = matplotlib.transforms.Bbox(np.array([
            [x_mid-buff, y_mid-buff], [x_mid+buff, y_mid+buff]
        ]))
        plt.figure(figsize=(6, 6))
        plt.plot(
            xs, ys, 'ro', markersize=2, clip_box=bounds, scalex=True,
            scaley=True
        )
        plt.show()


if __name__ == '__main__':
    import sys
    problem = TSPProblem(sys.argv[1])
    problem.show()
