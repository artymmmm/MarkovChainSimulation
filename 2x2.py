import numpy as np


# для 2x2
def launch():
    transitionPr = np.array([[0.7, 0.3], [0.1, 0.9]])
    state = 0
    for i in range(9):
        prob = np.random.choice(2, p=transitionPr[state])
        state = prob
    return state


# математика
transitionPr = np.array([[0.7, 0.3], [0.1, 0.9]])
initialPr = np.array([1, 0])
transitionPrToPower = np.linalg.matrix_power(transitionPr, 10)
finalPr = np.matmul(initialPr, transitionPrToPower)
print(finalPr)

# имит модел
oneCount = 0
launchCount = 0
for i in range(100000):
    resultState = launch()
    if resultState == 1:
        oneCount += 1
    launchCount += 1
modelPr = oneCount / launchCount
print(modelPr)