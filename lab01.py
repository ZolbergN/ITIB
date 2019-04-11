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
    y = [0]*16
    errors = np.ones(len(x))
    sumError = []

    k = 1
    while np.sum(errors) != 0:

        for i in range(0, len(x)):
            if func_activ_der == 1:
                net_y = net(w, x[i])
                y[i] = func_activ(net_y)
                delta = f[i] - y[i]

                for j in range(len(w)):
                    w[j] += eta * delta * x[i][j]
            else:
                net_y = net(w, x[i])
                y[i] = func_activ(net_y)
                delta = f[i] - y[i]

                for j in range(len(w)):
                    w[j] += eta * delta * x[i][j] * func_activ_der(y[i])

        errors = sum((f[i] ^ y[i]) for i in range(16))
        sumError.append(errors)

        print('%d y=%s, w=[%.2f, %.2f, %.2f, %.2f, %.2f], Error=%d' % (k, str(y), w[0], w[1], w[2], w[3], w[4], errors))
        k += 1

        if k > 50: return -1

    return k - 1, sumError

# Определение позиции набора в таблице истинности
def position(num_of_vec):
    return num_of_vec[4] + 2 * num_of_vec[3] + 4 * num_of_vec[2] + 8 * num_of_vec[1]

# Обучение НС с пороговой и сигмоидальной функциями методом полного перебора
def training_brute_force(f, x, func_activ, func_activ_der, num_of_vec, flag):
    n = 4
    w = np.zeros(n+1)
    eta = 0.3
    errors = np.ones(len(num_of_vec))
    y = [0]*len(num_of_vec)
    sumError = []

    for vectors in num_of_vec:
        k = 1
        while np.sum(errors) != 0:

            for i in range(len(num_of_vec)):
                if func_activ_der == 1:
                    net_y = net(w, num_of_vec[i])
                    y[i] = func_activ(net_y)
                    delta = f[i] - y[i]

                    for j in range(len(w)):
                        w[j] += eta * delta * num_of_vec[i][j]
                else:
                    net_y = net(w, num_of_vec[i])
                    y[i] = func_activ(net_y)
                    delta = f[i] - y[i]

                    for j in range(len(w)):
                        w[j] += eta * delta * num_of_vec[i][j] * func_activ_der(y[i])

            errors = sum((f[i] ^ y[i]) for i in range(len(num_of_vec)))
            sumError.append(errors)

            if flag:
                print('%d y=%s, w=[%.2f, s%.2f, %.2f, %.2f, %.2f], Error=%d' % (
                    k - 1, str(y), w[0], w[1], w[2], w[3], w[4], errors))

            k += 1

            if k >= 10: return -1

        if np.sum(errors) == 0:
            error = test_func_step(f, x, w, func_activ)

            if error == 0:
                if flag:
                    plt.plot(sumError, 'ro-')
                    plt.grid(True)
                    plt.show()
                return k - 1



    return 0

def test_func_step(f, x, w, func_activ):
    y = [func_activ(net(w, x[i])) for i in range(16)]

    err = sum((f[j] ^ y[j] for j in range(16)))

    return err

# Функция для предоставления необходимых данных алгоритму обучения
def step_brute_force_command():
    act_func = step
    der_act_func = 1

    # Перебираем всевозможное количество векторов, которые будут использоваться в обучении
    for i in range(2, 16):
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
    for i in range(2, 16):
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

# Функция для предоставления необходимых данных алгоритму обучения и построение графика
def step_command():
    k, errors = training_mode(func_init(initialize()), initialize(), step, 1)
    print('\nОбучилась за %d эпох' % k)

    plt.plot(errors, 'bo-')
    plt.grid(True)
    plt.show()


# Функция для предоставления необходимых данных алгоритму обучения и построение графика
def sigmoid_command():
    k, errors = training_mode(func_init(initialize()), initialize(), sigmoid, der_sigmoid)
    print('\nОбучилась за %d эпох' % k)

    plt.plot(errors, 'bo-')
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
