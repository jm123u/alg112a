import random

def hillClimbing (f,q, h= 0.1):
    fail_count = 0
    while fail_count < max_iterations:
        current_value, new_p, new_value = f(p), neighbor(p,h), f(neighbor(p, h))
        if new_value >= current_value:
            p = new_p
            print('p=', p, 'f(p)=', new_value)
        fail_count += 1
    return (p, current_value)

def neighbor(p,h) :
    new_p = [x + random.uniform(-h, h) for x in p]
    return new_p

def f(p):
    x,y,z = p
    return -1 * (x ** 2 + y ** 2 + z ** 2)

res = hillClimbing(f, [2,1,3])
print("Optimal soulution: x=", result[0][0],  "y =", result[0][1], "z =", result[0][2], "objective function value =", result[1])