import numpy as np
import math

def sigmoid(net):
    return (1 - np.exp(-net)) / (1 + np.exp(-net))

def sigmoid_derivative(net):
    f = (1 - np.exp(-net)) / (1 + np.exp(-net))

    return 0.5 * (1 - f ** 2)

class Neuron(object):
    _N = 1
    _J = 2
    _M = 1

    _epsilon = 1
    _etta = 1

    _X = [1.0, 3.0]
    _Xj = [0.0]*(_J + 1)
    _Y = [0.0]*(_M)
    _t = -0.4
    _net_hidden = [0.0]*(_J)
    _net_output = [0.0]*(_M)

    _era = 0
    _w = [[0.5, 0.5, -0.3, -0.7], [0.5, 0.3, -0.8]]

    _delta1 = [0.0]*(_J)
    _delta2 = [0.0]*(_M)

    # Алгоритм обучения
    def alghoritm(self):
        while(self._epsilon > 0.0001):
            print("ЭПОХА №", self._era)

            self._Xj[0] = 1.0
            # Считаем net скрытого слоя и выходные сигналы скрытого слоя
            for j in range(self._J):
                self._net_hidden[j] = self._w[0][j] * self._X[0] + self._w[0][j + 2] * self._X[1]
                self._Xj[j + 1] = sigmoid(self._net_hidden[j])

            # Считаем net выходного слоя и выходные сигналы выходного слоя
            for m in range(self._M):
                self._net_output[m] = self._w[m + 1][m] * self._Xj[m]
                for i in range(self._J):
                    self._net_output[m] += self._w[m + 1][i + 1] * self._Xj[i + 1]
                self._Y[m] = sigmoid(self._net_output[m])

                # Считаем ошибку на выходе
                self._delta2[m] = sigmoid_derivative(self._net_output[m]) * (self._t - self._Y[m])
                # Считаем суммарную среднеквадратичную ошибку
                self._epsilon = math.sqrt(sum([(self._t - self._Y[index]) ** 2 for index in range(self._M)]))

                # Считаем ошибку на скрытом слое
                for i in range(self._J):
                    self._delta1[i] = sigmoid_derivative(self._net_hidden[i]) * self._w[1][m + 1] * self._delta2[m]

                # Корректируем веса, начиная с выходного слоя
                for j in range(self._J + self._M):
                    self._w[1][j] += self._etta * self._delta2[m] * self._Xj[j]

            # Корректируем веса скрытого слоя
            for j in range(self._J):
                self._w[0][j] += self._etta * self._delta1[j] * self._X[self._N - 1]
                self._w[0][j + 2] += self._etta * self._delta1[j] * self._X[self._N]
            
            print("Комбинированный вход скрытого слоя: ", np.round(self._net_hidden, 4))
            print("Выходной сигнал скрытого слоя: ", np.round(self._Xj, 4))
            print("Комбинированный вход выходного слоя: ", np.round(self._net_output, 4))
            print("Выходной сигнал выходного слоя: ", np.round(self._Y, 4))
            print("Ошибка выходного слоя: ", self._delta2)
            print("Ошибка скрытого слоя: ", self._delta1)
            print("\n")
            print("Скорректированыые веса:", end='')
            for i in self._w:
                print(np.around(i, 4), end='')
            print('')

            print("-----------------------------------------------")
            print("Номер эпохи k: %d; Суммарная ошибка E(k): %.6f" % (self._era, self._epsilon))
            print("Выходной вектор y: ", np.round(self._Y, 4))
            print("-----------------------------------------------")

            self._era += 1

            print('\n')

obj = Neuron()

if __name__ == '__main__':
    obj.alghoritm()
