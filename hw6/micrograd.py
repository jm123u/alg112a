import micrograd as mg

def target_function(p):
    x, y, z = p
    return x**2 + y**2 + z**2

def grad_descent(f,p,step=0.01, epsilon=1e-4, max_iterations=1000):
    p = [mg.Value(val) for val in p]
    
    for i in range(max_iterations):
        loss = f(p)
        loss.backward()
        
        for j in range(len(p)):
            p[j].data -= step * p[j].grad.data
            p[j].zero_grad()
            
        if loss.data < epsilon:
            break
        
    res = [val.data for val in p]
    return res

if __name__ == '__main__':
    initial_point = [6, 8, 9]
    res = grad_descent(target_function, initial_point, step=0.01, epsilon=1e-6)
    print("Mini point:", res)
    print("Obj function value:", target_function(res))