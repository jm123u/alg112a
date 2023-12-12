import numpy as np
from micrograd.engine import Value
from micrograd.nn import sequential
from micrograd.optim import SGD

def target_function(p):
    return sum(pi ** 2 for pi in p)

if __name__ == '__main__':
    initial_point = [6, 8, 9]
    p = [Value(pi) for pi in initial_point]

    model = sequential(Dense(3, 1, bias=False))

    optimizer = SGD([p])

    for epoch in range(1000):
        loss = target_function(p)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    res = [pi.data for pi in p]

    print("Minimum point:", res)
    print("Objective function value:", target_function(res))
