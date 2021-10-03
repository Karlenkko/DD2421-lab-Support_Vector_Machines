import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

polynomial_kernel_power = 3
radial_kernel_sigma = 2
slack_C = None

def generate_data():
    global N
    N = 40

    np.random.seed(100)
    classA = np.concatenate((np.random.randn(int(N / 4), 2) * 0.2 + [1.5, 0.5], np.random.randn(int(N / 4), 2) * 0.2 + [-1.5, 0.5]))
    classB = np.random.randn(int(N / 2), 2) * 0.2 + [0.0, -0.5]
    inputs = np.concatenate((classA, classB))
    targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
    N = inputs.shape[0]  # Number of rows ( samples )
    permute = list(range(inputs.shape[0]))
    random.shuffle(permute)

    inputs = inputs[permute, :]
    targets = targets[permute]
    def plot_train():
        plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.' )
        plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.' )
        plt.axis('equal' )  # Force same s c a l e on both axes

    plot_train()
    return inputs, targets


def polynomial_kernel(x1, x2):
    # polynomial kernel
    return (np.dot(x1, x2) + 1) ** polynomial_kernel_power


def linear_kernel(x1, x2):
    return np.dot(x1, x2) + 1


def radial_kernel(x, y):
    diff = np.subtract(x, y)
    return math.exp((-np.dot(diff, diff)) / (2 * radial_kernel_sigma * radial_kernel_sigma))


def objective(alpha):
    sum_alpha = sum(alpha)
    w = 0.5 * np.dot(np.array(alpha).reshape(-1, 1), np.array(alpha).reshape(1, -1)) * p
    return sum(sum(w)) - sum_alpha
    # for i in range(alpha.shape[0]):
    #     for j in range(alpha.shape[0]):
    #         w += 0.5 * alpha[i] * alpha[j] * t[i] * t[j] * kernel(x[i], x[j])
    # return w - sum_alpha


def calculate_p(t, kernel, x):
    global p
    p = np.dot(np.array(t).reshape(-1, 1), np.array(t).reshape(1, -1))
    x2 = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            x2[i][j] = kernel(x[i], x[j])
    p *= x2


def zerofun(alpha):
    return sum(np.dot(np.array(alpha).reshape(1, -1), np.array(y).reshape(-1, 1)))



def low_filter(i):
    if i > 1e-5:
        return i
    else:
        return 0.0


def calculate_b(alpha, kernel):
    b = 0.0
    s = None
    index = 0
    for i in range(N):
        if alpha[i] > 0.0:
            s = x[i]
            index = i
            break
    for i in range(N):
        b += alpha[i] * y[i] * kernel(s, x[i])
    b -= y[index]
    return b


def indicator(alpha, b, point, kernel):
    ind = 0.0
    for i in range(N):
        ind += alpha[i] * y[i] * kernel(point, x[i])
    ind -= b
    return ind

if __name__ == '__main__':
    global x, y
    # ker = radial_kernel
    ker = polynomial_kernel
    # ker = linear_kernel
    
    x, y = generate_data()
    calculate_p(y, ker, x)
    bound = [(0, slack_C) for b in range(N)]
    ret = minimize(objective, np.zeros(N), bounds=bound, constraints={'type': 'eq', 'fun': zerofun})
    if not ret["success"]:
        exit(1)
    alpha = list(map(lambda x: low_filter(x), ret["x"]))
    b = calculate_b(alpha, ker)
    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace(-4, 4)
    grid=np.array([[indicator(alpha, b, [x, y], ker) for x in xgrid ] for y in ygrid])
    plt.contour( xgrid , ygrid , grid ,(-1.0, 0.0, 1.0 ) , colors=('red', 'black', 'blue') , linewidths=(1 , 3 , 1))
    plt.show()

