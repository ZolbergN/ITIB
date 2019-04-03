import math
import numpy as np
import matplotlib.pyplot as plt

N = 20

def initialize(xlist):
    ylist = [math.sin(x - 1) for x in xlist]

    return ylist

def Net(x, w):
    return sum([w_i * x_i for w_i, x_i in zip(w, x)])

def training_mode(xlist, p, eta, T):
    M = 30
    xx = [0]*20
    w = [0 for i in range(p)]

    for q in range(p):
        xx[q] = xlist[q]

    era = 0

    while(era < M):

        for l in range(p, N):
            xx[l] = Net(xlist[l - p:l-1], w)
            err = xlist[l] - xx[l]
            for k in range(p):
                w[k] += eta * err * xlist[l - p + k]
            print("w = ", w)
            print(err)

        era += 1

        E = sum((math.sqrt((xlist[i] - xx[i]) ** 2) for i in range(N)))

        print('E: ', E)

        print("era = ", era)
        print("xx = ", xx)

    plot_func(xlist, xx, T)

    return xx

def plot_func(xlist, xx, TT):
    fig, ax1 = plt.subplots()
    ax1.plot(TT, xlist, 'ro-')
    ax1.plot(TT, xx, 'bo-')

    plt.xlabel("T")
    plt.ylabel("X")

    print('x: ',xx)
    print('xlist: [', xlist)

    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    p = 4
    N = 20
    a = -2
    b = 2
    dx = 0.2
    net = 0.0
    eta = 0.5

    a_learn = b
    b_learn = 2*b - a

    T = np.arange(a, b, dx)

    xlist = initialize(T)

    print('\n')

    training_mode(xlist,  p, eta, T)
