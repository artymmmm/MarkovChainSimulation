import numpy as np


# для NxN

# функция одной симуляции
# параметры: матрица переходных вероятностей,
# количество шагов, изначальное состояниеразмер матрицы
def launch(probability_matrix, steps, first_state, matrix_size):
    state = first_state
    for i in range(steps):
        prob = np.random.choice(matrix_size, p=probability_matrix[state])
        state = prob
    return state


# пользователь вводит исходные данные
matrixSize = int(input('Введите количество состояний: '))
firstState = int(input('Введите начальное состояние: '))
lastState = int(input('Введите конечное состояние: '))
steps = int(input('Введите количество шагов: '))
simulations = int(input('Введите количество симуляций: '))

# пользователь вводить матрицу переходных вероятностей
probabilityList = []
for i in range(matrixSize):
    row = []
    for j in range(matrixSize):
        probability = float(input(
            'Введите переходную вероятность из {0} в {1}: '.format(i, j)))
        row.append(probability)
    probabilityList.append(row)
probabilityMatrix = np.array(probabilityList)

# моделирование
goodEndCount = 0
launchCount = 0
for i in range(simulations):
    resultState = launch(probabilityMatrix, steps, firstState, matrixSize)
    if resultState == lastState:
        goodEndCount += 1
    launchCount += 1
modelPr = goodEndCount / launchCount

# вычисление по формуле Pr = Pr0 * (ProbMatrix**steps)
initialPr = np.zeros(matrixSize)
initialPr[firstState] = 1  # начальный вектор
transitionPrToPower = np.linalg.matrix_power(probabilityMatrix,
                                             steps)  # степень
finalPr = np.matmul(initialPr,
                    transitionPrToPower)  # умножение вектора на матрицу
finalStatePr = finalPr[lastState]

print('''\nВероятность попадания в состояние {0} на шаге {1}:\n1) {2}, 
полученная с помощью вычислений\n2) {3}, полученная с помощью 
моделирования'''.format(lastState, steps, finalStatePr, modelPr))
