import numpy as np
import json

class Kohonen:
    def __init__(self, filePath):
        self.districts = {'Центральный административный округ': [55.750000, 37.616670],
                          'Северный административный округ': [55.838384, 37.525765],
                          'Северо-Восточный административный округ': [55.863894, 37.620923],
                          'Восточный административный округ': [55.787710, 37.775631],
                          'Юго-Восточный административный округ': [55.692019, 37.754583],
                          'Южный административный округ': [55.610906, 37.681479],
                          'Юго-Западный административный округ': [55.662735, 37.576178],
                          'Западный административный округ': [55.728003, 37.443533],
                          'Северо-Западный административный округ': [55.829370, 37.451546],
                          'Зеленоградский административный округ': [55.987583, 37.194250],
                          'Троицкий административный округ': [55.355771, 37.146990],
                          'Новомосковский административный округ': [55.558121, 37.370724]}

        self.patternNames = []
        self.patternCoordinates = []
        self.test = []

        self.parse(filePath)

    def parse(self, filePath):
        with open(filePath, encoding="utf8") as file:
            data = json.load(file)
            for i in data:
                self.patternNames.append(i['FullName'])
                self.patternCoordinates.append(i['geoData']['coordinates'][0])
                self.test.append(i['ObjectAddress'][0]['AdmArea'])

    def distance(self, x, center):
        distance = x[0]
        for i in range(1, len(x)):
            distance += center[i - 1] * x[i]

        return distance

    def alghoritm(self):
        errors = 0
        weights = []
        for clasterCoordinates in self.districts.values():
            weight = []
            weight.append(-0.5 * (clasterCoordinates[0] ** 2 + clasterCoordinates[1] ** 2))
            weight.append(clasterCoordinates[0])
            weight.append(clasterCoordinates[1])
            weights.append(weight)

        for l in range(len(self.patternCoordinates)):
            neuronsWinners = []
            coordinates = []
            coordinates.append(self.patternCoordinates[l][1])
            coordinates.append(self.patternCoordinates[l][0])

            for j in range(len(weights)):
                neuronsWinners.append(self.distance(weights[j], coordinates))
            print(f'Hospital Index: {l} | Coordinates: [{self.patternCoordinates[l][0]}]'
                  f'[{self.patternCoordinates[l][1]}] | '
                  f'Real District: {self.test[l]} -> Actual District: '
                  f'{list(self.districts)[neuronsWinners.index(max(neuronsWinners))]} | '
                  f'{self.test[l] == list(self.districts)[neuronsWinners.index(max(neuronsWinners))]}')

            if (self.test[l] == list(self.districts)[neuronsWinners.index(max(neuronsWinners))]) == False:
                errors += 1

        return errors

obj = Kohonen('hospital.json')

if __name__ == '__main__':
    errors = obj.alghoritm()

    print(f"Процент ошибки: {np.round((errors / (len(obj.patternNames))) * 100, 1)}%")
