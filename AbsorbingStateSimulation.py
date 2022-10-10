import numpy as np
import statistics


def get_absorb_state(pr_matrix):
    '''
    Получение поглощающего состояния из матрицы вероятностей
    :param pr_matrix: матрица вероятностей
    :return: поглощающее состояние
    '''
    i = 0
    while i < pr_matrix.shape[0]:
        if int(pr_matrix[i][i]) == 1:
            return i
        i += 1
    raise ValueError("Отсутствует поглощающее состояние")
    return None


def get_coef_matrix(pr_matrix, absorb_st):
    '''
    Получение матрицы коэффициентов
    :param pr_matrix: матрица вероятностей
    :param absorb_st: поглощающее состояние
    :return: матрица коэффициентов
    '''
    matrix = []
    for row in range(pr_matrix.shape[0]):
        if row == absorb_st:
            continue

        coefs = []
        for column in range(pr_matrix.shape[1]):
            if column == absorb_st:
                continue
            if row == column:
                coefs.append(pr_matrix[row, column] - 1)
            else:
                coefs.append(pr_matrix[row, column])
        matrix.append(coefs)
    return np.array(matrix)


def slau(pr_matrix, absorb_st):
    '''
    Получение среднего времени перехода в поглощающее состояние
    с помощью решения СЛАУ
    :param pr_matrix: матрица состояний
    :param absorb_st: поглощающее состояние
    :return: среднее время перехода в поглощающее состояние
    '''
    coef_matrix = get_coef_matrix(pr_matrix, absorb_st)
    result_vector = np.ones((coef_matrix.shape[0], 1))
    result_vector = np.negative(result_vector) # вектор правых частей
    inverse_matrix = np.linalg.inv(coef_matrix) # обратная матрица
    x = np.matmul(inverse_matrix, result_vector) # умножение матриц
    return x.transpose()[0]


def launch(pr_matrix, absorb_st, first_st):
    '''
    Один эксперимент имитационного моделирования
    :param pr_matrix: матрица вероятностей
    :param absorb_st: поглощающее состояние
    :param first_state: начальное состояние
    :return: количество шагов до попадания в поглощающее состояние
    '''
    state = first_st
    step = 0
    while state != absorb_st:
        state = np.random.choice(pr_matrix.shape[0], p=pr_matrix[state])
        step += 1
    return step


def simulation_modelling(pr_matrix, absorb_st, simulations):
    '''
    Получение среднего времени перехода в поглощающее состояние
    с помощью ИМ
    :param pr_matrix: матрица вероятностей
    :param absorb_st: поглощающее состояние
    :param simulations: количество экспериментов
    :return: среднее время перехода в поглощающее состояние
    '''
    sim_results = []
    for first_state in range(pr_matrix.shape[0]):
        if first_state == absorb_st:
            continue
        state_results = []
        for i in range(simulations):
            state_results.append(
                launch(pr_matrix, absorb_st, first_state))  # эксперименты
        mean_time = statistics.fmean(state_results)
        sim_results.append(mean_time)
    return sim_results


def get_result(pr_matrix, simulations):
    '''
    Получение результатов двумя способами
    :param pr_matrix: матрица вероятностей
    :param simulations: количество экспериментов
    :return: None
    '''

    # проверка матрицы
    for row in range(pr_matrix.shape[0]):
        if np.sum(pr_matrix[row]) != 1:
            raise ValueError("Сумма вероятностей не равна 1")

    absorb_state = get_absorb_state(pr_matrix)
    slau_res = slau(pr_matrix, absorb_state).tolist()
    sim_res = simulation_modelling(pr_matrix, absorb_state, simulations)
    print("""Среднее время попадания в поглощающее состояние:
    1) {0}, полученное с помощью решения СЛАУ;
    2) {1}, полученное с помощью ИМ""".format(slau_res, sim_res))


probability_matrix = np.array([[0.3, 0.5, 0.2],
                               [0.8, 0.1, 0.1],
                               [0, 0, 1]])

get_result(probability_matrix, 10000)
