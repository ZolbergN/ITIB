import numpy as np
import math
import matplotlib.pyplot as plt

class Neuron(object):
    _plotX = []  # Список для координат по X
    _plotY = []  # Список для координат по Y

    _M = 1000

    _learnFunction = [] # Список для обучающейся функции
    _mainFunction = []  # Список для функции по условию

    _N = 20
    _a = -2
    _b = 2

    _windowSize = 4
    _w = np.zeros(_windowSize + 1)

    _epsilon = 1
    _eta = 0.3
    _delta = 0
    _k = 0

    _era = []
    _error = []
    # Считаем net
    def net(self, windowSize, x, w):
        net = (sum(w[i + 1] * x[i] for i in range(windowSize)))

        return net

    # Получаем интервал с шагом
    def getX(self, a, b):
        vectorX = []

        dx = (b - a) / self._N

        for i in range(self._N):
            vectorX.append(a + i * dx)

        return vectorX

    # Получаем значения функции на интервале
    def getY(self, a, b):
        vectorY = []
        vectorX = self.getX(a, b)

        for i in range(self._N):
            vectorY.append(math.sin(vectorX[i] - 1))

        return vectorY

    # Алгоритм обучения
    def training_mode(self):
        self._learnFunction = [0]*20
        self._mainFunction = self.getY(self._a, self._b)

        self._w = np.zeros(self._windowSize + 1)

        while self._k < self._M:
            for q in range(self._windowSize):
                self._learnFunction[q] = self._mainFunction[q]

            for i in range(self._windowSize, self._N):
                self._learnFunction[i] = self.net(self._windowSize, self._mainFunction[i - self._windowSize: i], self._w)
                self._delta = self._mainFunction[i] - self._learnFunction[i]

                for j in range(self._windowSize):
                    self._w[j + 1] += self._eta * self._delta * self._mainFunction[i-self._windowSize + j]

            self._epsilon = sum((self._mainFunction[index] - self._learnFunction[index]) ** 2 for index in range(self._N))
            self._epsilon = math.sqrt(self._epsilon)

            self._era.append(self._k)
            self._error.append(self._epsilon)

            self._k += 1

    def assingment(self, command):
        self._epsilon = 1
        self._mainFunction = self.getY(self._a, self._b)

        if command == 'eta':
            self.training_mode()
            self._plotY.append(self._epsilon)
            self._plotX.append(self._eta)
        elif command == 'p':
            self.training_mode()
            self._plotY.append(self._epsilon)
            self._plotX.append(self._windowSize)
        elif command == 'era':
            self.training_mode()
            plt.plot(obj._era, obj._error, 'bo-')
            plt.grid(True)
            plt.show()

        print("Веса: ", np.round(self._w, 4))
        print('Среднеквадратичная ошибка: ', self._epsilon)

obj = Neuron()

if __name__ == '__main__':
    commands = 'Введите команду:' \
               '\n     eta      --- зависимость среднеквадратичной ошибки от нормы обучения' \
               '\n     p        --- зависимость среднеквадратичной ошибки от размера окна' \
               '\n     era      --- зависимость среднеквадратичной ошибки от количества эпох' \
               '\n' \
               '\n     exit     --- выход из программы'

    print(commands)

    while True:
        command = input()

        if command == 'eta':
            obj._plotX = []
            obj._plotY = []
            obj._windowSize = 4
            for i in range(1, 10):
                obj._eta = float(i / 10)
                print("Норма обучения: ", obj._eta)
                obj._k = 0
                obj._M = 1000
                obj.assingment(command)

                print("Обучилась за %d эпох" % obj._k)
                print("\n")

            plt.plot(obj._plotX, obj._plotY, 'ro-')
            plt.grid(True)
            plt.show()

        elif command == 'p':
            obj._plotX = []
            obj._plotY = []
            obj._eta = 0.3

            for i in range(2, 12):
                obj._windowSize = i
                print("Размер окна: ", obj._windowSize)
                obj._k = 0
                obj._M = 1000
                obj.assingment(command)

                print("Обучилась за %d эпох" % obj._k)
                print("\n")

            plt.plot(obj._plotX, obj._plotY, 'ro-')
            plt.grid(True)
            plt.show()

        elif command == 'era':
            obj._plotX = []
            obj._plotY = []
            obj._era = []
            obj._error = []

            obj._eta = 0.3
            obj._windowSize = 4
            obj._M = 25
            obj._k = 0

            obj.assingment(command)

            print("Размер окна: ", obj._windowSize)
            print("Норма обучения: ", obj._eta)
            print("Обучилась за %d эпох" % obj._M)

        elif command == 'exit':
            break

        else:
            print("Введена не корректная команда")
