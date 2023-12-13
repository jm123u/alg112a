# 期末總結
```
課程名稱：演算法
授課老師： 陳鍾誠 教授
用途： 期初至期末作業總結
```

## Hw1 : 費氏數列迴圈版
```
# 自行編寫
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
```

## Hw2 power2n 四種實作方法
```
# 自行學習理解 編寫 
package main

import (
	"fmt"
)

// Method 1: Using the exponentiation operator
func power2nMethod1(n int) int {
	return 1 << n
}

// Method 2a: Using recursion
func power2nMethod2a(n int, memo map[int]int) int {
	if n == 0 {
		return 1
	}
	if val, ok := memo[n]; ok {
		return val
	}
	memo[n] = 2 * power2nMethod2a(n-1, memo)
	return memo[n]
}

// Method 2b: Using recursion
func power2nMethod2b(n int) int {
	if n == 0 {
		return 1
	}
	return 2 * power2nMethod2b(n-1)
}

// Method 3: Using recursion with memoization
func power2nMethod3(n int, memo map[int]int) int {
	if n == 0 {
		return 1
	}
	if val, ok := memo[n]; ok {
		return val
	}
	memo[n] = 2 * power2nMethod3(n-1, memo)
	return memo[n]
}

func main() {
	n := 5

	// Method 1
	result1 := power2nMethod1(n)
	fmt.Printf("Method 1: 2^%d = %d\n", n, result1)

	// Method 2a
	memo2a := make(map[int]int)
	result2a := power2nMethod2a(n, memo2a)
	fmt.Printf("Method 2a: 2^%d = %d\n", n, result2a)

	// Method 2b
	result2b := power2nMethod2b(n)
	fmt.Printf("Method 2b: 2^%d = %d\n", n, result2b)

	// Method 3
	memo3 := make(map[int]int)
	result3 := power2nMethod3(n, memo3)
	fmt.Printf("Method 3: 2^%d = %d\n", n, result3)
}


#以下為測試程序
package main

import (
	"testing"
)

func TestPower2nMethods(t *testing.T) {
	n := 5
	expected := 32 // 2^5 = 32

	// Test Method 1
	result1 := power2nMethod1(n)
	if result1 != expected {
		t.Errorf("Method 1: Expected 2^%d = %d, but got %d", n, expected, result1)
	}

	// Test Method 2a
	memo2a := make(map[int]int)
	result2a := power2nMethod2a(n, memo2a)
	if result2a != expected {
		t.Errorf("Method 2a: Expected 2^%d = %d, but got %d", n, expected, result2a)
	}

	// Test Method 2b
	result2b := power2nMethod2b(n)
	if result2b != expected {
		t.Errorf("Method 2b: Expected 2^%d = %d, but got %d", n, expected, result2b)
	}

	// Test Method 3
	memo3 := make(map[int]int)
	result3 := power2nMethod3(n, memo3)
	if result3 != expected {
		t.Errorf("Method 3: Expected 2^%d = %d, but got %d", n, expected, result3)
	}
}
```

## hw3 寫出可以列舉出所有排列的程序
```
/* Writing using backtracking
Using too large a number may take a long time to calculate. Learning GO for the first time.*/
# 自行學習 進行編寫 使用回溯法

package main

import (
	"fmt"
)

func generatePermutations(n int) [][]int {
	result := make([][]int, 0)
	permutation := make([]int, n)
	used := make([]bool, n)

	var backtrack func(int)
	backtrack = func(idx int) {
		if idx == n{
			temp := make([]int, n)
			copy(temp, permutation)
			result = append(result, temp)
			return
		}

		for i := 0; i < n; i++{
			if ! used[i] {
				 used[i] = true
				 permutation[idx] = i
				 backtrack(idx + 1)
				 used[i] = false
				}
		}
	}

	backtrack(0)
	return result
}

func main() {
	n := 4   //Modifiable Requires a number
	permutation :=generatePermutations(n)
	
	for _, p := range permutation {
		fmt.Println(p)
	}
}

# 以下為c 自行編寫
#include<stdio.h>
 void generateTruthTable(int n) {
    int numRows = 1 << n;
    for(int i =0; i < numRows; i++) {
        printf("[");
        for(int j = 0; j<n; j++ ) {
            printf("%d", (i >> j) & 1);
            if( j< n -1){
                printf(", ");
            }
        }
        printf("]\n");
    }
 }

 int main () {
    int n = 3;
    generateTruthTable(n);
    return 0;
 }

```

## Hw4 : 求解方程式
```
/* 
create： 2023.9.30
with : hw3
use: newtonraphson, bisection, regulafalsi
A simple attempt to learn the Go language.
*/
// 自行學習 看hello 算法給我靈感 自行編寫

package main

import (
	"fmt"
	"math"
)

// Newton's iterative
func newtonRaphson() {
	x0 := 0.0 
	tolerance := 1e-6

	for {
		f := x0*x0 - 3*x0 + 1
		df := 2*x0 -3
		x1 := x0 - f/df
		
		if math.Abs(x1-x0) < tolerance {
			fmt.Printf("Newton's Method: Approximate root: %.6f\n", x1)
			break
		}

		x0 = x1
	}
}

// dichotomy
func bisection() {
	a := -10.0
	b := 10.0
	tolerance := 1e-6

	for {
		c := (a + b) / 2
		f := c*c - 3*c + 1

		if math.Abs(f) < tolerance {
			fmt.Printf("Bisection Method: Approximate root: %.6f\n", c)
			break
		}

		if f > 0 {
			b = c
		} else {
			a = c
		}
	}
}


func regulaFalsi() {
	a := -10.0
	b := 10.0
	tolerance := 1e-6

	for {
		fa := a*a - 3*a + 1
		fb := b*b - 3*b + 1
		c := (a*fb - b*fa) / (fb - fa)
		f := c*c - 3*c +1

		if math.Abs(f) < tolerance {
			fmt.Printf("Regula Falsi Method: Approcimate root: %.6f \n",c)
			break
		}

		if f > 0{
			b = c
		} else {
			a = c 
		}
	}
}

func main () {
	newtonRaphson()
	bisection()
	regulaFalsi()
}

# 暴力法
package main

import (
	"fmt"
	"math"
)

func main() {
	epsilon := 1e-4
	x := -100.0

	for {
		fx := x*x - 3*x + 1

		if math.Abs(fx) < epsilon {
			fmt.Printf("x of the equation (brute force): %.6f\n", x)
			break
		}

		x += 0.001
	}
}

```

## Hw5: 爬山演算法程式可以找任何向量函數的山頂
```
# 老師程序碼做參考做出改動 其他為自己編寫
import random

def hillClimbing(f, p, max_iterations=10000, h=0.0001): # h越小越精確 為 領域範圍參數
    fail_count = 0
    while fail_count < max_iterations:
        current_value, new_p, new_value = f(p), neighbor(p, h), f(neighbor(p, h))
        if new_value >= current_value:
            p = new_p
            print('p=', p, 'f(p)=', new_value)
        fail_count += 1
    return (p, current_value)

def neighbor(p, h):
    new_p = [x + random.uniform(-h, h) for x in p]
    return new_p

def f(p):
    x, y, z = p
    return -1 * (x ** 2 + y ** 2 + z ** 2)

res = hillClimbing(f, [2, 1, 3])
print("Optimal solution: x =", res[0][0], "y =", res[0][1], "z =", res[0][2], "Objective function value =", res[1])
```

## Hw6 寫一個梯度下降法程式可以找任何向量函數的谷底
```
# 自行理解編寫
import numpy as np

def df(f, p, k, step=0.001):
    p1 = p.copy()
    p1[k] = p[k] + step
    return (f(p1) - f(p)) / step

def grad(f, p, step=0.01, epsilon=1e-6):
    p = np.array(p)
    
    while True:
        gradient = np.array([df(f, p, k, step) for k in range(len(p))])
        p_new = p - step * gradient
        
        if np.linalg.norm(gradient) < epsilon:
            return p_new
        
        p = p_new
        
def target_function(p):
    return np.sum(np.square(p))

if __name__ == '__main__':
    initial_point = [6, 8, 9]
    res = grad(target_function, initial_point, step=0.01, epsilon=1e-6)
    print("Minimum point:", res)
    print("Objective function value:", target_function(res))
```

## Hw7 : 根據習題六的程式，改用 micrograd 的反傳遞算法算梯度
```
# 由chatgpt 生成 已理解
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
```

## Hw8 : 選一位圖靈獎得主，詳細說明他的得獎原因
https://hackmd.io/ZicIhloCQu6K0iOE8TukPg?view=

## Hw9:搜尋法求解下列問題其中的一個
```
#自行理解 编写 八个皇后问题
#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define N 8

bool is_safe(int board[N], int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || abs(board[i] - col) == abs(i - row)) {
            return false;
        }
    }
    return true;
}

void print_solution(int board[N]) {
    for (int i = 0; i < N; i++, puts(""))
        for (int j = 0; j < N; j++)
            putchar(board[i] == j ? 'Q' : '.');
    puts("");
}

void solve_eight_queens(int board[N], int row) {
    if (row == N) {
        print_solution(board);
        return;
    }
    for (int col = 0; col < N; col++) {
        if (is_safe(board, row, col)) {
            board[row] = col;
            solve_eight_queens(board, row + 1);
            board[row] = -1;
        }
    }
}

int main() {
    int board[N];
    for (int i = 0; i < N; i++) {
        board[i] = -1;
    }
    solve_eight_queens(board, 0);

    return 0;
}
```

## Hw10:寫一個程式可以求解 n 次多項式
*x^5 + 1 =0*
*x^8+3x^2+1=0*
```
# 自行學習編寫
package main

import (
	"fmt"
	"math"
)

func equation(x float64) float64 {
	return math.Pow(x, 8) + 3*math.Pow(x, 2) + 1
}

func main() {
	epsilon := 1e-6
	lower, upper := -10.0, 10.0

	for math.Abs(upper-lower) > epsilon {
		mid := (lower + upper) / 2
		if equation(mid)*equation(lower) < 0 {
			upper = mid
		} else {
			lower = mid
		}
	}

	root := (lower + upper) / 2
	fmt.Printf("Root: %f\n", root)
	fmt.Printf("Value at Root: %f\n", equation(root))
}
```

## Hw challenge 用遞迴寫最小編輯距離 (紅利）
```
#自行編寫
def editDistance(a, b):
    m, n = len(a), len(b)

    dp = [[0] * (n + 1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1) :
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp [i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i-1][j], dp[i-1][j-1])
    
    return dp[m][n]

a = "ATGCAATCCC"
b = "ATGATCCG"

result = editDistance(a, b)
print(f"editDistance({a}, {b}) = {result}")

#自行學習編寫 換種方式進行編寫
import json

def editDistanceRecursive(a, b):
    memo = {}

    def helper(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if i == 0:
            result = j
        elif j == 0:
            result = i
        else:
            cost = 0 if a[i - 1] == b[j - 1] else 1

            result = min(
                helper(i-1, j) + 1,
                helper(i, j-1) + 1,
                helper(i-1, j-1) + cost
            )

        memo[(i, j)] = result
        return result

    distance = helper(len(a), len(b))
    return {'d': distance, 'm': memo}

def dump(m):
    for key, value in m.items():
        print(json.dumps({str(key): value}))

a = 'ATGCAATCCC'
b = 'ATGATCCG'

result = editDistanceRecursive(a, b)
print(f'editDistanceRecursive({a}, {b}) = {result["d"]}')
print('====m======\n')
dump(result['m'])
```

