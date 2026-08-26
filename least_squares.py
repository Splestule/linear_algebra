import numpy as np
import matplotlib.pyplot as plt
from solver import (swap_rows, scale_row, add_multiple_of_row,
                    elimination, backsubstitution, inversion, RREF, null_space_special, four_subspaces)

line = [2, 5]
cubic = [1, -2, 3, -1]

points = 50
rng = np.random.default_rng(54512)

b_line = []
A_line = []
xs_line = []

for i in range(points):
    x = rng.random()*10
    y = line[0]*x + line[1]
    b_line.append(y+rng.normal())
    A_line.append([1, x])
    xs_line.append(x)

plt.scatter(np.array(xs_line), np.array(b_line))
ys_plot_line = []
for x in [0, 10]:
    ys_plot_line.append(line[0]*x + line[1])
plt.plot(np.array([0, 10]), np.array(ys_plot_line))
plt.show()
    
b_cubic = []
A_cubic = []
xs_cubic = []

for i in range(points):
    x = rng.random()*3
    y = cubic[0]*x**3 + cubic[1]*x**2 + cubic[2]*x + cubic[3]
    b_cubic.append(y+rng.normal())
    A_cubic.append([1, x, x**2, x**3])
    xs_cubic.append(x)
    
plt.figure()
plt.scatter(np.array(xs_cubic), np.array(b_cubic))
ys_plot_cubic = []
for x in np.linspace(0, 3, 200):
    ys_plot_cubic.append(cubic[0]*x**3 + cubic[1]*x**2 + cubic[2]*x + cubic[3])
plt.plot(np.array(np.linspace(0, 3, 200)), np.array(ys_plot_cubic))
plt.show()

def gradient_descent(A, b, iterations=500, learning_rate=0.001):
    parameters = np.zeros(len(A[0]))
    mean_squared_loss = []
    for i in range(iterations):
        loss = []
        for m in range(len(A)):
            local_loss = 0
            for j in range(len(parameters)):
                local_loss += parameters[j]*A[m][j]
            loss.append(local_loss - b[m])
        for j in range(len(parameters)):
            gradient = 0
            for p in range(len(A)):
                gradient += loss[p]*A[p][j]
            parameters[j] -= 2*(gradient/len(A))*learning_rate
        mean_squared_loss.append(np.average(np.array(loss)**2))
    plt.figure()
    plt.plot(mean_squared_loss)
    plt.show()
    return mean_squared_loss, parameters

def least_squares(A, b):
    A = np.array(A)
    b = np.array(b)
    x_hat = inversion(A.T@A)@A.T@b
    p = A@x_hat
    return p, x_hat

mean_squared_loss, parameters = gradient_descent(A_cubic, b_cubic, iterations=5000000)
print("loss: ", mean_squared_loss[-1], " parameters: ", parameters)

p, x_hat = least_squares(A_cubic, b_cubic)
print("loss: ", np.mean((b_cubic-p)**2), " parameters: ", x_hat)