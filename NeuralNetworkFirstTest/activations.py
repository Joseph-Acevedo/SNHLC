import numpy as np

def sigmoid_activation(x):
    if(np.abs(x.any()) < 60):
        return 1 / (1 + np.exp(-x))
    else:
        return 1 / (1 + np.exp(-60))

def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sin_activation(x):
    return np.sin(x)

def sin_derivative(x):
    return np.cos(x)

def gauss_activation(x):
    return np.exp(-x ** 2)

def gauss_derivative(x):
    return -2 * x * np.exp(-x ** 2)

def relu_activation(x):
    return x if x.any() > 0 else 0

def relu_derivative(x):
    return 1 if x.any() > 0 else 0

def softplus_activation(x):
    if(np.abs(x.any()) < 60):
        return np.log(1 + np.exp(x))
    else:
        return np.log(1 + np.exp(60))

def softplus_derivative(x):
    if(np.abs(x.any()) < 60):
        return 0.2 * 1 / (1 + np.exp(-x))
    else:
        return 0.2 * 1 / (1 + np.exp(-60))

def identity_activation(x):
    return x

def identity_derivative(x):
    return 1

def clamped_activation(x):
    if(x.any() < -1):
        return -1
    elif(np.abs(x.any()) < 1):
        return x
    elif(x.any() > 1):
        return 1
    else:
        return x

def clamped_derivative(x):
    if(x.any() < -1):
        return 0
    elif(np.abs(x.any()) < 1):
        return 1
    elif(x.any() > 1):
        return 0
    else:
        return 1

def inv_activation(x):
    try:
        x = 1 / x
    except ArithmeticError:
        return 0
    else:
        return x

def inv_derivative(x):
    try:
        x = -1 / x ** 2
    except ArithmeticError:
        return 0
    except RuntimeWarning:
        return 0
    else:
        return 1
        
def log_activation(x):
    if(x.any() > 0.0000001):
        return np.log(x)
    else:
        return np.log(0.0000001)

def log_derivative(x):
    try:
        x = 1 / x
    except ArithmeticError:
        return 0
    else:
        return 1

def exp_activation(x):
    if(x.any() < 60):
        return np.exp(x)
    else:
        return np.exp(60)

def exp_derivative(x):
    if(x.any() < 60):
        return np.exp(x)
    else:
        return np.exp(60)

def abs_activation(x):
    return np.abs(x)

def abs_derivative(x):
    if(x.any() < 0):
        return -1
    elif(x.any() > 0):
        return 1
    else:
        return 0

def hat_activation(x):
    return np.max(0, 1 - np.abs(x.any()))

def hat_derivative(x):
    if(np.abs(x.any()) > 1):
        return 0
    elif(x.any() < 0):
        return 1
    elif(x.any() > 0):
        return -1
    else:
        return 0

def square_activation(x):
    return x ** 2

def square_derivative(x):
    return 2 * x

def cube_activation(x):
    return x ** 3

def cube_derivative(x):
    return 3 * x ** 2
