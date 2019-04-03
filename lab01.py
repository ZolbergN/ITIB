from builtins import input
from itertools import combinations
from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt

def net(w,x):
    return sum(w[i] * x[i] for i in range(5))

def step(net):
    if net >= 0:
        return 1
    else:
        return 0

def sigmoid(net):
    if 0.5 * (net / (1 + abs(net)) + 1) >= 0.5:
        return 1
    else:
        return 0
def der_sigmoid(net):
    f = 0.5 * (net / (1 + abs(net)) + 1)
    return 0.5 / ((1 + abs(f)) ** 2)

def training_mode(f, x, func_activ, func_activ_der):
    n = 4
    w = np.zeros(n+1)
    eta = 0.3

    y = [func_activ(net(w, x[i])) for i in range(16)]

    err = sum(f[i] ^ y[i] for i in range(16))

    errors = [err]

    print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], err=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], err))

    k = 1
    while err != 0:
        delta = list(f[i] - y[i] for i in range(16))

        for j in range(5):
            if func_activ_der == 1:
                w[j] += sum(eta * delta[i] * x[i][j] for i in range(16))
            else:
                w[j] += sum(eta * delta[i] * x[i][j] * func_activ_der(net(w, x[j])) for i in range(16))

        y = [func_activ(net(w, x[i])) for i in range(16)]

        err = sum(f[i] ^ y[i] for i in range(16))
        errors.append(err)

        print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], err=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], err))

        k += 1

        if k >= 50: return -1

    return k, errors

def index(num_of_vec):
    return num_of_vec[4] + 2 * num_of_vec[3] + 4 * num_of_vec[2] + 8 * num_of_vec[1]

def training_brute_force(f, x, func_activ, func_activ_der, num_of_vec, flag):
    n = 4
    w = np.zeros(n+1)

    eta = 0.3

    y = [func_activ(net(w, x[i])) for i in range(16)]
    err = sum(f[i] ^ y[i] for i in range(16))
    errors = [err]

    if flag:
        print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], err=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], err))

    k = 1
    while err != 0:
        delta = list((f[i] - y[i] for i in range(16)))

        for j in range(5):
            if func_activ_der == 1:
                w[j] += sum(eta * delta[index(num_of_vec[i])] * num_of_vec[i][j] for i in range(len(num_of_vec)))
            else:
                w[j] += sum(eta * delta[index(num_of_vec[i])] * num_of_vec[i][j] * func_activ_der(net(w, num_of_vec[i]))
                            for i in range(len(num_of_vec)))

        y = [func_activ(net(w, x[i])) for i in range(16)]

        err = sum((f[i] ^ y[i] for i in range(16)))
        errors.append(err)

        if flag:
            print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], err=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], err))

        k += 1

        if k >= 50: return -1

    plt.plot(errors)
    plt.grid(True)
    plt.show()

    return k

def step_brute_force_command():
    act_func = step
    der_act_func = 1

    for i in range(16):
        all_combinations = list(combinations(x, i))

        print('Перебор из %d векторов...' % i)

        for num_of_vec in all_combinations:
            flag = 0
            count = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)

            if count > 0:
                print('Наборы: %s' % str(num_of_vec))

                flag = 1
                k = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)
                print('\nОбучилась за %d эпох' % k)

                break
        if flag == 1: break

def sigmoid_brute_force_command():
    act_func = sigmoid
    der_act_func = der_sigmoid

    for i in range(16):
        all_combinations = list(combinations(x, i))

        print('Перебор из %d векторов...' % i)

        for num_of_vec in all_combinations:
            flag = 0
            count = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)

            if count > 0:
                print('Наборы: %s' % str(num_of_vec))

                flag = 1
                k = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)

                print('\nОбучилась за %d эпох' % k)

                break
        if flag == 1: break

def step_command():
    k, _ = training_mode(func_init(initialize()), initialize(), step, 1)
    print('\nОбучилась за %d эпох' % k)

def sigmoid_command():
    k, _ = training_mode(func_init(initialize()), initialize(), sigmoid, der_sigmoid)
    print('\nОбучилась за %d эпох' % k)

def step_plot():
    act_func = step
    der_act_func = 1

    _, errors = training_mode(f, x, act_func, der_act_func)
    plt.plot(errors)
    plt.grid(True)
    plt.show()

def sigmoid_plot():
    act_func = sigmoid
    der_act_func = der_sigmoid

    _, errors = training_mode(f, x, act_func, der_act_func)
    plt.plot(errors)
    plt.grid(True)
    plt.show()

def initialize():
    n = 4
    X = []
    i = 0
    while i < 2**n:
        x = list(format(i, f'0{n}b'))
        x = [int(s) for s in x]
        X.append(x)
        i += 1

    for element in X:
        element.insert(0, 1)

    print(X)

    return X

def func_init(X):
    F = list()
    target_function = []
    for element in X:
        act1 = element[1] and element[2]
        act2 = act1 or element[3]
        act3 = act2 or element[4]
        target_function.append(int(act3))
    for i in range(len(X)):
        F.append(target_function[i])

    print('Target Function =', F)
    return F

if __name__ == '__main__':

    x = initialize()
    f = func_init(x)

    commands = 'Введите команду:' \
    '\n     step            --- обучение нейронной сети и построение графика ошибок для пороговой функции' \
    '\n     sigmoid         --- обучение нейронной сети и построение графика ошибок для сигмоидальной функции' \
    '\n     step-brute      --- обучение нейронной сети для пороговой функции полным перебором' \
    '\n     sigmoid-brute   --- обучение нейронной сети для сигмоидальной функции полным перебором' \
    '\n' \
    '\n     exit        --- выход из программы'

    print(commands)

    true = 1

    while true == 1:
        command = input()

        if command == 'step':
            step_command()
            step_plot()
        elif command == 'sigmoid':
            sigmoid_command()
            sigmoid_plot()
        elif command == 'step-brute':
            step_brute_force_command()
        elif command == 'sigmoid-brute':
            sigmoid_brute_force_command()
        elif command == 'exit':
            true = 0
        else:
            print("Некорректная команда")

