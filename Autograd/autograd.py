from math import exp, log

DEBUG = 1


class Node(object):
    def __init__(self, value):
        if isinstance(value, Node):
            self.value = value.value
        else:
            self.value = value
        self.grad = 0

    def mSelf(self, data):
        return data if isinstance(data, Node) else Node(data)

    def __add__(self, data):
        return Add(self, self.mSelf(data))

    def __radd__(self, data):
        return Add(self.mSelf(data), self)

    def __sub__(self, data):
        return Sub(self, self.mSelf(data))

    def __rsub__(self, data):
        return Sub(self.mSelf(data), self)

    def __mul__(self, data):
        return Mul(self, self.mSelf(data))

    def __rmul__(self, data):
        return Mul(self.mSelf(data), self)

    def __truediv__(self, data):
        return Div(self, self.mSelf(data))

    def __rtruediv__(self, data):
        return Div(self.mSelf(data), self)

    def __pow__(self, data):
        return Pow(self, self.mSelf(data))

    def __rpow__(self, data):
        return Pow(self.mSelf(data), self)

    def __neg__(self):
        return Sub(Node(0), self)

    def __repr__(self):
        return "Node {}".format(self.value)


class Op(Node):
    def __init__(self, left, right):
        self.left = left if isinstance(left, Node) else Node(left)
        self.right = right if isinstance(right, Node) else Node(right)
        self.value = self.forward()
        self.op = "Op" if not hasattr(self, "op") else self.op
        super().__init__(self.value)
        # print(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __repr__(self):
        return "Op [{}]\t, [{}\t, {}\t] => {}\t".format(
            self.op, self.left.value, self.right.value, self.forward()
        )


class Add(Op):
    def __init__(self, left, right):
        self.op = "Add"
        super().__init__(left, right)

    def forward(self):
        return self.left.value + self.right.value

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += 1 * backgrad
        self.right.grad += 1 * backgrad
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)
        if isinstance(self.right, Op):
            self.right.backward(self.right.grad)


class Sub(Op):
    def __init__(self, left, right):
        self.op = "Sub"
        super().__init__(left, right)

    def forward(self):
        return self.left.value - self.right.value

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += 1 * backgrad
        self.right.grad -= 1 * backgrad
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)
        if isinstance(self.right, Op):
            self.right.backward(self.right.grad)


class Mul(Op):
    def __init__(self, left, right):
        self.op = "Mul"
        super().__init__(left, right)

    def forward(self):
        return self.left.value * self.right.value

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += self.right.value * backgrad
        self.right.grad += self.left.value * backgrad
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)
        if isinstance(self.right, Op):
            self.right.backward(self.right.grad)


class Div(Op):
    def __init__(self, left, right):
        self.op = "Div"
        super().__init__(left, right)

    def forward(self):
        return self.left.value / self.right.value

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += backgrad / self.right.value
        self.right.grad += (
            backgrad * (-1 * self.left.value) / self.right.value ** 2
        )
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)
        if isinstance(self.right, Op):
            self.right.backward(self.right.grad)


class Pow(Op):
    def __init__(self, left, right):
        self.op = "Pow"
        super().__init__(left, right)

    def forward(self):
        return self.left.value ** self.right.value

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += (
            backgrad
            * self.right.value
            * self.left.value ** (self.right.value - 1)
        )
        self.right.grad += backgrad * log(self.left.value) * self.value
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)
        if isinstance(self.right, Op):
            self.right.backward(self.right.grad)


class Exp(Op):
    def __init__(self, left, right=None):
        self.op = "Exp"
        super().__init__(left, right)

    def forward(self):
        return exp(self.left.value)

    def backward(self, backgrad=None):
        if DEBUG:
            print(self)
        if backgrad is None:
            backgrad = 1
        self.left.grad += backgrad * self.value
        if isinstance(self.left, Op):
            self.left.backward(self.left.grad)


if __name__ == "__main__":
    a = Node(3)
    b = Node(2)
    f = (2 * a + b ** a) * b
    f.backward()
    print(a.grad, b.grad)
