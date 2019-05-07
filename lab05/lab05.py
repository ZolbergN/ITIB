from os.path import isfile

import numpy as np
import os

values = {1: '1', -1: '0'}
use = {'1': 1, '0': -1}

SIDE_HEIGHT = 5
SIDE_WIDTH = 3

known_shapes_dir = 'known_shapes'
test_shapes_path = 'test_shapes/testShapes.txt'

SIZE = 32

def function_activation(net, y):
    if net > 0:
        return 1
    elif net == 0:
        return y
    else:
        return -1

class NeuronHopfield:
    def __init__(self, height, width):
        self.drawsteps = False
        self.shapes = []
        self.n = height * width
        self.height = height
        self.width = width
        self.w = np.array([np.zeros(self.n) for _ in range(self.n)])
        self.max_iter = 300
        self.current_iter = 0

    def training_mode(self, x):
        self.shapes.append(x)

        for i in range(self.n):
            for j in range(self.n):
                if(i == j):
                    self.w[i][j] = 0
                else:
                    self.w[i][j] += x[i] * x[j]

    def net(self, x):
        for i in range(self.n):
            net_y = sum([self.w[j][i] * x[j] for j in range(self.n)])
            y = function_activation(net_y, x[i])
            if y != x[i] and y != 0:
                print(f"Neuron {i} : {x[i]} -> {y}")
                x[i] = y
            if x not in self.shapes:
                return (f"False ", x)

        return (f"Success. Training completed in iterations", x)

    def parse(self, directory):
        shapes_files = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if isfile(path):
                shapes_files.append(path)
        shapes = []
        for path in shapes_files:
            shape = self.parse_shape(path)
            shapes.append(shape)
        return shapes

    def parse_shape(self, path):
        with open(path) as f:
            shape = f.read(SIZE)
            shape = shape.replace("\n", "")
            shape = shape.replace("\r", "")

            shapes = []
            for c in shape:
                shapes.append(use[c])
            if len(shapes) != (SIDE_HEIGHT * SIDE_WIDTH):
                raise Exception("Shape size must be %gx%g" % (SIDE_WIDTH, SIDE_HEIGHT))
            return shapes

    def printshape(self, obraz, heigth, widht):
        for_print = "".join([values[a] for a in obraz])
        for i in range(heigth):
            print(for_print[i * widht: i * widht + widht])
        print('')

n = NeuronHopfield(SIDE_HEIGHT, SIDE_WIDTH)

if __name__ == '__main__':
    shapes = n.parse(known_shapes_dir)
    shape = n.parse_shape(test_shapes_path)

    print("Known Shapes:")
    for o in shapes:
        n.printshape(o, SIDE_HEIGHT, SIDE_WIDTH)

    print("Teaching...")
    for o in shapes:
        n.training_mode(o)

    print("Modified shape:")
    n.printshape(shape, SIDE_HEIGHT, SIDE_WIDTH)

    print("Shape definition...")
    answer, reshape = n.net(shape)
    print(answer)
    n.printshape(reshape, SIDE_HEIGHT, SIDE_WIDTH)
