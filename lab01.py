from builtins import input
from itertools import combinations
from matplotlib.pyplot import *
import numpy as np
import matplotlib.pyplot as plt

#Функция, которая считает net
def net(w,x):
    return sum(w[i] * x[i] for i in range(5))

# Пороговая функция
def step(net):
    if net >= 0:
        return 1
    else:
        return 0

# Сигмоидальная функция
def sigmoid(net):
    if 0.5 * (net / (1 + abs(net)) + 1) >= 0.5:
        return 1
    else:
        return 0

# Производная от сигмоидальной функции
def der_sigmoid(net):
    f = 0.5 * (net / (1 + abs(net)) + 1)
    return 0.5 / ((1 + abs(f)) ** 2)

# Алгоритм для обучения НС с пороговой и сигмоидальной функциями
def training_mode(f, x, func_activ, func_activ_der):
    n = 4
    w = np.zeros(n+1)
    eta = 0.3

    # Посчитаем реальный выход перед обучением
    y = [func_activ(net(w, x[i])) for i in range(16)]

    # Посчитаем суммарную ошибку перед обучением
    err = sum(f[i] ^ y[i] for i in range(16))

    errors = [err]

    print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], Error=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], err))

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

        print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], Error=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], err))

        k += 1

        if k >= 50: return -1

    return k - 1, errors

# Определение позиции набора в таблице истинности
def position(num_of_vec):
    return num_of_vec[4] + 2 * num_of_vec[3] + 4 * num_of_vec[2] + 8 * num_of_vec[1]

# Обучение НС с пороговой и сигмоидальной функциями методом полного перебора
def training_brute_force(f, x, func_activ, func_activ_der, num_of_vec, flag):
    n = 4
    w = np.zeros(n+1)
    eta = 0.3

    # Посчитаем реальный выход перед обучением
    y = [func_activ(net(w, x[i])) for i in range(16)]

    # Посчитаем суммарную ошибку перед обучением
    err = sum(f[i] ^ y[i] for i in range(16))

    errors = [err]

    if flag:
        print('0 y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], Error=%d' % (str(y), w[0], w[1], w[2], w[3], w[4], err))

    k = 1
    while err != 0:
        delta = list((f[i] - y[i] for i in range(16)))

        for j in range(5):
            if func_activ_der == 1:
                w[j] += sum(eta * delta[position(num_of_vec[i])] * num_of_vec[i][j] for i in range(len(num_of_vec)))
            else:
                w[j] += sum(eta * delta[position(num_of_vec[i])] * num_of_vec[i][j] * func_activ_der(net(w, num_of_vec[i]))
                            for i in range(len(num_of_vec)))

        y = [func_activ(net(w, x[i])) for i in range(16)]

        err = sum((f[i] ^ y[i] for i in range(16)))
        errors.append(err)

        if flag:
            print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], Error=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], err))

        k += 1

        if k >= 10: return -1

    if flag:
        plt.plot(errors)
        plt.grid(True)
        plt.show()

    return k - 1

# Функция для предоставления необходимых данных алгоритму обучения
def step_brute_force_command():
    act_func = step
    der_act_func = 1

    # Перебираем всевозможное количество векторов, которые будут использоваться в обучении
    for i in range(1, 16):
        all_combinations = list(combinations(x, i))

        print('Перебор из %d векторов...' % i)

        for num_of_vec in all_combinations:
            # Используем флаг для того, чтобы понять, что на НС обучилась
            flag = 0
            count = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)

            if count > 0:
                print('Наборы: %s' % str(num_of_vec))

                flag = 1
                k = training_brute_force(f, x, act_func, der_act_func, num_of_vec, flag)
                print('\nОбучилась за %d эпох' % k)

                break
        if flag == 1: break

# Функция для предоставления необходимых данных алгоритму обучения
def sigmoid_brute_force_command():
    act_func = sigmoid
    der_act_func = der_sigmoid

    # Перебираем всевозможное количество векторов, которые будут использоваться в обучении
    for i in range(1, 16):
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

# Функция для предоставления необходимых данных алгоритму обучения и построение графика
def step_command():
    k, errors = training_mode(func_init(initialize()), initialize(), step, 1)
    print('\nОбучилась за %d эпох' % k)

    plt.plot(errors)
    plt.grid(True)
    plt.show()


# Функция для предоставления необходимых данных алгоритму обучения и построение графика
def sigmoid_command():
    k, errors = training_mode(func_init(initialize()), initialize(), sigmoid, der_sigmoid)
    print('\nОбучилась за %d эпох' % k)

    plt.plot(errors)
    plt.grid(True)
    plt.show()

# Строим таблицу истинности
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

# Записываем нашу функцию
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
        elif command == 'sigmoid':
            sigmoid_command()
        elif command == 'step-brute':
            step_brute_force_command()
        elif command == 'sigmoid-brute':
            sigmoid_brute_force_command()
        elif command == 'exit':
            true = 0
        else:
            print("Некорректная команда")
