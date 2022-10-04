import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


matplotlib.use('MacOSX')


def get_coef_matrix(pr_matrix):
    '''
    Получение матрицы коэффициентов из матрицы вероятностей
    :param pr_matrix: матрица вероятностей
    :return: матрица коэффициентов (NxN)
    '''
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
    '''
    Получение столбца правых частей из матрицы вероятностей
    :param matrix: матрица вероятностей
    :return: столбец правых частей (вектор Nx1)
    '''
    vector = np.ndarray((matrix.shape[0], 1))
    for i in range(matrix.shape[0] - 1):
        vector[i, 0] = 0
    vector[-1, 0] = 1
    return vector


def get_sim_modelling_res(res_dict_list, i, size):
    '''
    :param res_dict_list: список словарей со значениями вероятностей
    :param i: словарь, который требуется преобразовать
    :param size: количество строк матрицы вероятностей
    :return: список вероятностей
    '''
    res_list = []
    for l in range(size):
        if l in res_dict_list[i].keys():
            res_list.append(res_dict_list[i][l])
        else:
            res_list.append(0)
    return res_list


def math_power(pr_matrix, first_state):
    '''
    Расчет результата возведением исходной матрицы в степень
    :param pr_matrix: матрица вероятностей
    :param first_state: начальное состояние
    :return: первая строка матрицы, возведенной в степень; значения дельты, степень
    '''
    power_matrix = np.copy(pr_matrix)
    matrix_states = [power_matrix[first_state]]
    delta_res = []
    delta = 1
    i = 1
    while delta > 0.0001:
        power_matrix = np.matmul(power_matrix, pr_matrix)
        matrix_states.append(power_matrix[first_state])
        sub_res = np.subtract(power_matrix[0], power_matrix[1])
        abs_res = np.absolute(sub_res)
        delta = np.mean(abs_res)
        delta_res.append(delta)
        i += 1
    return matrix_states, delta_res, i


def math_linear_system(c_matrix, vector):
    '''
    Расчет результата решением СЛАУ
    :param c_matrix: матрица коэффициентов
    :param vector: столбец правых частей
    :return: вектор значений (1xN)
    '''
    """
    Расчет результата решением СЛАУ
    Parameter: матрица коэффициентов, столбец правых частей
    Return: вектор значений (1xN)
    """
    inverse_matrix = np.linalg.inv(c_matrix)
    x = np.matmul(inverse_matrix, vector)
    return x.transpose()[0]


def launch(pr_matrix, steps, first_state):
    '''
    Один экспериментов имитационного моделирования
    :param pr_matrix: матрица вероятностей
    :param steps: количество шагов
    :param first_state: начальное состояние
    :return: конечное состояние системы
    '''
    state = first_state
    # цикл по шагам, где выбирается следующее состояние в соответствии со строкой
    # матрицы вероятностей
    for i in range(steps):
        state = np.random.choice(pr_matrix.shape[0], p=pr_matrix[state])
    return state


def simulation_modeling(pr_matrix, steps, first_state, simulations):
    '''
    Расчет результата имитационным моделированием
    :param pr_matrix: матрица вероятностей
    :param steps: количество шагов
    :param first_state: начальное состояние
    :param simulations: количество экспериментов
    :return: словарь с вероятностями
    '''
    endStates = []
    for i in range(simulations):
        endStates.append(launch(pr_matrix, steps, first_state))  # эксперименты
    results = {i: endStates.count(i) for i in
               endStates}  # количество для каждого конечного состояния
    for i in results:
        results[i] = results[i] / simulations  # расчет вероятностей
    return results


def get_results(pr_matrix, first_state, simulations):
    '''
    Получение всех результатов, вывод и построение графика
    :param pr_matrix: матрица вероятностей
    :param first_state: начальное состояние
    :param simulations: количество экспериментов
    :return: none
    '''
    simulation_modeling_res = []
    coef_matrix = get_coef_matrix(pr_matrix)  # матрица коэффициентов
    result_vector = get_result_vector(pr_matrix)  # столбец правых значений

    math_power_res = math_power(pr_matrix, first_state)

    for i in range(math_power_res[2]):
        simulation_modeling_res.append(
            simulation_modeling(pr_matrix, i + 1, first_state,
                                simulations))  # моделирование
    math_linear_system_res = math_linear_system(coef_matrix,
                                                result_vector)  # СЛАУ

    simulation_modeling_res_list = get_sim_modelling_res(
        simulation_modeling_res, -1, pr_matrix.shape[0])

    # вывод результатов
    print('''Стационарное распределение:
        1) {0}, полученное решением СЛАУ;
        2) {1}, полученное возведением матрицы вероятностей в степень;
        3) {2}, полученное имитационным моделированием.'''.format(
        math_linear_system_res, math_power_res[0][-1],
        simulation_modeling_res_list))

    # график
    sim_model_res_list = [get_sim_modelling_res(simulation_modeling_res, i, pr_matrix.shape[0])
                          for i in range(len(simulation_modeling_res))]

    for k in range(pr_matrix.shape[0]):
        plt.plot(list(range(1, len(math_power_res[0]) + 1)),
                 [math_power_res[0][i][k] for i in
                  range(len(math_power_res[0]))], '--o', label='Состояние '
                                                               '{0} (степень)'.format(k))
        plt.plot(list(range(1, len(simulation_modeling_res) + 1)),
                 [sim_model_res_list[i][k] for i in range(len(simulation_modeling_res))], '--o',
                 label='Состояние {0} (ИМ)'.format(k))
    plt.xticks(list(range(1, len(simulation_modeling_res) + 1)))
    plt.title('Зависимость вероятности от количества шагов')
    plt.legend()
    plt.show()


prob_matrix = np.array([[0.7, 0.3],
                        [0.1, 0.9]])  # матрица вероятностей
get_results(prob_matrix, 0, 1000)
