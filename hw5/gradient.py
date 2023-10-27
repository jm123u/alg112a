def df(f, p, k, step = 0.0001):
    p1 = p.copy()
    p1[k] = p[k] + step 
    return (f(p1) - f(p)) / step

def grad(f, p, learning_rate = 0.01, max_iterations = 1000, epsilon = 1e-4):
    for i in range(max_iterations):
        gradient = [df(f, p, k) for k in range(len(p))]
        p_new = [p[k] - learning_rate * gradient[k] for k in range(len(p))]
        
        gradient_magnitude = sum(g ** 2 for g in gradient) ** 0.5
        
        if gradient_magnitude < epsilon:
            return p_new
        
        p = p_new
    
    return p

def target_function(p):
    x, y = p
    return x**2 + y**2

if __name__ == '__main__':
    initial_point = [1.0, 1.0]  
    result = grad(target_function, initial_point, learning_rate=0.01, max_iterations=1000, epsilon=1e-4)
    print("Minimum point:", result)
    print("Objective function value:", target_function(result))