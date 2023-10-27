import numpy as np

def df (f, p, k, step=0.01):
    p1 = p.copy()
    p1[k] = p[k] / step

def grad(f, p, step=0.01, epsilon=1e-6):
    p = np.array(p)
    
    while True:
        gradient = np.array([df(f, p, k, step) for k in range(len(p))])
        p_new = p - step * gradient
        
        if np.linalg.norm(gradient) < epsilon:
            return p_new
        
        p = p_new
        
def target_function(p):
    return sum(x**2 for x in p)

if __name__ == '__main__':
    initial_point = [2, 3, 4]
    res = grad(target_function,initial_point,step=0.01, epsilon=1e-4)
    print("mini poirt:",res)
    print("obj function value:", target_function(res))