import numpy as np


class Layer:
    def __init__(self):
        pass

    @property
    def need_update(self):
        return False


class FC(Layer):
    def __init__(self, W, b, lr, delta, decay, epoch_drop):
        self.W = W.copy()
        self.b = b.copy()
        self.rW = np.zeros_like(self.W)
        self.rb = np.zeros_like(self.b)
        self.alpha_0 = lr
        self.delta = delta
        self.decay = decay
        self.epoch_drop = epoch_drop
        self.count = 0

    def forward(self, x):
        self.x = x.copy()
        self.m, self.n = x.shape
        return np.dot(self.x, self.W) + self.b

    def backprop(self, back_grad):
        self.gW = np.dot(self.x.T, back_grad)
        self.gb = np.dot(np.ones(self.m), back_grad)
        self.rW += self.gW ** 2
        self.rb += self.gb ** 2
        self.grad = np.dot(back_grad, self.W.T)
        return self.grad

    def l_rate(self):
        lrate = self.alpha_0 * \
            (self.decay ** (np.floor((1 + self.count) / self.epoch_drop)))
        self.count += 1
        return lrate

    def update(self):
        alpha = self.l_rate()
        dW = -alpha / (self.delta + np.sqrt(self.rW)) * self.gW
        db = -alpha / (self.delta + np.sqrt(self.rb)) * self.gb
        self.W += dW
        self.b += db

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
        self.y = y.copy()
        self.m, self.n = self.x.shape
        self.denom = np.sum(np.exp(x), axis=1).reshape((-1, 1))
        self.softmax = np.exp(x) / self.denom
        loss = 0
        for i in range(self.m):
            loss -= np.log(self.softmax[i, y[i]])
            # this line can be also written as:
            # loss = sum(k from 1 to n (the category count))[-y_k*log(y_hat_k)]
            # we can see, if the y_k = 1 and y_hat_k = 1(nearly),
            # this means the model, gives the right answer,
            # so the loss of this term will be 0,
            # means that no loss on this term,
            # what's more, if y_hat_k is a way far from 1, the loss of this term
            # will be large, so the loss will punish the parameter in the Net
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
