import numpy as np


class Layer:
    def __init__(self):
        pass

    @property
    def need_update(self):
        return False


class FC(Layer):
    def __init__(self, W, b, lr, decay, epoch_drop, l2=0):
        self.W = W.copy()
        self.b = b.copy()
        self.alpha_0 = lr
        self.decay = decay
        self.epoch_drop = epoch_drop
        self.l2 = l2
        self.count = 0

    def forward(self, x):
        self.x = x.copy()
        self.m, self.n = x.shape
        return np.dot(self.x, self.W) + self.b

    def backprop(self, back_grad):
        self.grad_W = np.dot(self.x.T, back_grad) + self.l2 * self.W
        self.grad_b = np.dot(np.ones(self.m), back_grad)
        self.grad = np.dot(back_grad, self.W.T)
        return self.grad

    def l_rate(self):
        lrate = self.alpha_0 * \
            (self.decay ** (np.floor((1 + self.count) / self.epoch_drop)))
        self.count += 1
        return lrate

    def update(self):
        lr = self.l_rate()
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

    @property
    def need_update(self):
        return True


class Sigmoid(Layer):
    def forward(self, x):
        self.x = x.copy()
        self.sig_res = 1 / (1 + np.exp(-x))
        return self.sig_res

    def backprop(self, back_grad):
        grad = back_grad * self.sig_res * (1 - self.sig_res)
        return grad


class Relu(Layer):
    def forward(self, x):
        self.x = x.copy()
        return np.maximum(x, 0)

    def backprop(self, back_grad):
        grad = back_grad.copy()
        grad[self.x < 0] = 0
        return grad


class Leaky_Relu(Layer):
    def forward(self, x):
        self.x = x.copy()
        return np.maximum(x, self.x * 0.01)

    def backprop(self, back_grad):
        grad = back_grad.copy()
        grad[self.x < 0] = grad[self.x < 0] * 0.01
        return grad


class Tanh(Layer):
    def forward(self, x):
        self.x = x.copy()
        self.tanh = np.tanh(x)
        return self.tanh

    def backprop(self, back_grad):
        grad = back_grad * (1 - self.tanh ** 2)
        return grad


class Arctan(Layer):
    def forward(self, x):
        self.x = x.copy()
        return np.arctan(self.x)

    def backprop(self, back_grad):
        grad = back_grad / (1 + self.x ** 2)
        return grad


class SoftPlus(Layer):
    def forward(self, x):
        self.x = x.copy()
        return np.log(1 + np.exp(self.x))

    def backprop(self, back_grad):
        grad = back_grad / (1 + np.exp(-self.x))
        return grad


class SoftSign(Layer):
    def forward(self, x):
        self.x = x.copy()
        return self.x / (1 + np.abs(self.x))

    def backprop(self, back_grad):
        grad = back_grad / (1 + np.abs(self.x) ** 2)
        return grad


class Softmax(Layer):
    def forward(self, x, y):
        self.x = (x.copy() - x.max(axis=1).reshape(-1, 1))
        # Avoiding overflow of exp(),
        # This operation doesn't change the output of CE
        self.y = y.copy()
        self.m, self.n = self.x.shape
        self.denom = np.sum(np.exp(x), axis=1).reshape((-1, 1))
        self.softmax = np.exp(x) / self.denom
        loss = 0
        for i in range(self.m):
            loss -= np.log(self.softmax[i, y[i]])
        return loss / self.m

    def dirac(self, a, b):
        return 1 if a == b else 0

    def backprop(self):
        grad = np.zeros([self.m, self.n])
        for i in range(self.m):
            for j in range(self.n):
                grad[i, j] = (self.softmax[i, j] -
                              self.dirac(j, self.y[i])) / self.m
        return grad


def get_act_func(layer_name):
    activation_function_dict = {
        "arctan": Arctan,
        "l_relu": Leaky_Relu,
        "relu": Relu,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "softplus": SoftPlus,
        "softsign": SoftSign
    }
    return activation_function_dict[layer_name]()
