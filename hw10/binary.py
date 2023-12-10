import random
import copy

def neighbor(f, x, h):
    x_new = x + random.uniform(-h, h)
    return f(x_new), x_new

def hill