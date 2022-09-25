import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
matplotlib.use('MacOSX')


def get_coef_matrix(pr_matrix):
    """
    Получение матрицы коэффициентов из матрицы вероятностей
    Parameter: матрица вероятностей
    Return: матрица коэффициентов (NxN)
    """
    new_matrix = []
    for j in range(pr_matrix.shape[1] - 1):  # столбцы
        row = []
        for i in range(pr_matrix.shape[0]):  # строки
            if i == j:
                row.append(pr_matrix[i, j] - 1)
            else:
                row.append(pr_matrix[i, j])
        new_matrix.append(row)
    new_matrix.append(np.ones(pr_matrix.shape[1]))
    return np.array(new_matrix)


def get_result_vector(matrix):
    """
    Получение столбца правых частей из матрицы вероятностей
    Parameter: матрица вероятностей
    Return: столбец правых частей (вектор Nx1)
    """
    vector = np.ndarray((matrix.shape[0], 1))
    for i in range(matrix.shape[0] - 1):
        vector[i, 0] = 0
    vector[-1, 0] = 1
    return vector


def math_power(pr_matrix, steps, first_state):
    """
    Расчет результата возведением исходной матрицы в степень
    Parameter: матрица вероятностей, количество шагов, начальное состояние
    Return: первая строка матрицы
    """
    pr_matrix = np.linalg.matrix_power(pr_matrix,
                                       steps)  # возведение в степень
    return pr_matrix[first_state]  # возврат строки матрицы


def math_linear_system(c_matrix, vector):
    """
    Расчет результата решением СЛАУ
    Parameter: матрица коэффициентов, столбец правых частей
    Return: вектор значений (1xN)
    """
    inverse_matrix = np.linalg.inv(c_matrix)
    x = np.matmul(inverse_matrix, vector)
    return x.transpose()[0]


def launch(pr_matrix, steps, first_state):
    """
    Один экспериментов имитационного моделирования
    Parameter: матрица вероятностей, количество шагов, начальное состояние
    Return: конечное состояние системы
    """
    state = first_state
    # цикл по шагам, где выбирается следующее состояние в соответствии со строкой
    # матрицы вероятностей
    for i in range(steps):
        state = np.random.choice(pr_matrix.shape[0], p=pr_matrix[state])
    return state


def simulation_modeling(pr_matrix, steps, first_state, simulations):
    """
    Расчет результата имитационным моделированием
    Parameter: матрица вероятностей, количество шагов, начальное состояние,
    количество экспериментов
    Return: словарь с вероятностями
    """
    endStates = []
    for i in range(simulations):
        endStates.append(launch(pr_matrix, steps, first_state))  # эксперименты
    results = {i: endStates.count(i) for i in
               endStates}  # количество для каждого конечного состояния
    for i in results:
        results[i] = results[i] / simulations  # расчет вероятностей
    return results


def get_results(pr_matrix, steps, first_state, last_state, simulations):
    """
    Получение всех результатов, вывод и построение графика
    Parameter: матрица вероятностей, количество шагов, начальное состояние,
    количество экспериментов
    Return: -
    """
    math_power_res = []
    simulation_modeling_res = []

    coef_matrix = get_coef_matrix(pr_matrix)  # матрица коэффициентов
    result_vector = get_result_vector(pr_matrix)  # столбец правых значений

    for i in range(steps):
        math_power_res.append(
            math_power(pr_matrix, i + 1, first_state))  # возведение в степень
        simulation_modeling_res.append(
            simulation_modeling(pr_matrix, i + 1, first_state,
                                simulations))  # моделирование
    math_linear_system_res = math_linear_system(coef_matrix,
                                                result_vector)  # СЛАУ

    # вывод результатов
    print('''Стационарное распределение:
    1) {0}, полученное решением СЛАУ;
    2) {1}, полученное возведением матрицы вероятностей в степень;
    3) {2}, полученное имитационным моделированием.'''.format(
        math_linear_system_res, math_power_res[-1],
        [simulation_modeling_res[-1][i] for i in range(len(simulation_modeling_res[-1]))]))

    # график
    plt.plot([math_power_res[i][last_state] for i in range(len(math_power_res))],
             list(range(1, steps + 1)), '--ro', label='Возведение в степень')
    plt.plot([simulation_modeling_res[i][last_state] for i in
              range(len(simulation_modeling_res))],
             list(range(1, steps + 1)), '--bo',
             label='Имитационное моделирование')
    plt.yticks(np.arange(0, steps + 1))
    plt.xlabel('Вероятность попадания из {0} в {1}'.format(first_state, last_state))
    plt.ylabel('Количество шагов')
    plt.legend()
    plt.show()


prob_matrix = np.array([[0.7, 0.1, 0.2], [0.1, 0.4, 0.5], [0.2, 0.7, 0.1]])  # матрица вероятностей

# количество шагов, начальное состояние, конечное состояние, количество экспериментов
get_results(
    prob_matrix, 20, 2, 0, 10000)
