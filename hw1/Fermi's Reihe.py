def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        prev, current = 0, 1
        for _ in range(2, n):
            next_fib = prev + current
            prev, current = current, next_fib
        return current

n = 3
result = fibonacci(n)
print(f"The {n}-th Fibonacci number is {result}")
