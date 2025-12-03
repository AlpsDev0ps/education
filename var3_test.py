import csv
from random import random

def mse(outputs, targets):
    error = 0

    for i, output in enumerate(outputs):
        error += (output - targets[i]) ** 2

    return error / len(outputs)


class LinearRegression:
    def __init__(self, features_num):
        # +1 for bias, bias is last weight
        self.weights = [random() * 2 - 1 for _ in range(features_num + 1)]  # нормализованные веса

    def forward(self, input_features):
        output = 0

        for i, feature in enumerate(input_features):
            output += self.weights[i] * feature

        output += self.weights[-1]

        return output

    def train(self, inp, output, target, samples_num, lr):
        for j in range(len(self.weights) - 1):
            self.weights[j] -= lr * (2 / samples_num) * (output - target) * inp[j]

        self.weights[-1] -= lr * (2 / samples_num) * (output - target)

    def fit(self, inputs, targets, epochs=100, lr=0.01):
        for epoch in range(epochs):
            outputs = []

            for i, inp in enumerate(inputs):
                output = self.forward(inp)
                outputs.append(output)

                self.train(inp, output, targets[i], len(inputs), lr)

            if epoch % 10 == 0:
                print(f"epoch: {epoch}, error: {mse(outputs, targets)}")

    def predict(self, inputs):
        predictions = []
        for inp in inputs:
            predictions.append(self.forward(inp))
        return predictions


def read_data_from_csv(filename, delimiter=';'):
    """Чтение данных из CSV файла"""
    inputs = []
    targets = []

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)  # читаем заголовки

        for row in reader:
            # Преобразуем значения в float
            values = [float(x) for x in row]

            # Y - это целевая переменная (первый столбец)
            targets.append(values[0])

            # X1-X4 - это признаки (остальные столбцы)
            inputs.append(values[1:])

    return inputs, targets, headers


if __name__ == '__main__':

    filename = 'input.csv'
    inputs, targets, headers = read_data_from_csv(filename)

    print(f"Загружено {len(inputs)} записей")
    print(f"Признаки: {headers[1:]} (всего {len(headers[1:])})")
    print(f"Целевая переменная: {headers[0]}")
    print(f"Пример данных: inputs[0] = {inputs[0]}, targets[0] = {targets[0]}")

    # Количество признаков = 4 (X1, X2, X3, X4)
    lr_model = LinearRegression(features_num=4)
    lr_model.fit(inputs, targets, epochs=1000, lr=0.001)

    print("\nОбучение завершено!")
    print(f"Веса модели: {lr_model.weights}")
    print(f"Последний вес (bias): {lr_model.weights[-1]}")

    # Делаем предсказания на обучающей выборке
    predictions = lr_model.predict(inputs[:5])
    print(f"\nПервые 5 предсказаний:")
    for i in range(5):
        print(f"Реальное значение: {targets[i]:.2f}, Предсказанное: {predictions[i]:.2f}")
