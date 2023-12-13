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

## hw 期中作業
演算法學習報告： 梯度消失&爆炸原因和部分解決辦法
https://hackmd.io/i2xnTUVvQ6-KXhjrMe2FNQ?view=

## hw 11  希爾伯特經圖靈到 NP-Complete 的故事寫下來

```
# NP-Complete定理
```
課程名稱：演算法
授課老師： 陳鍾誠
參考資料： Algorithms Illuminated (Part 4) Algorithms for NP-Hard Problems；https://ycc.idv.tw/algorithm-complexity-theory.html；https://zhuanlan.zhihu.com/p/235301347；https://zhuanlan.zhihu.com/p/433308577；
```

### 介紹這個問題之前我們先瞭解 *時間複雜度*
```
1.時間複雜度並不是表示一個程序解決問題需要花多少時間，而是當程序所處理的問題規模擴大后，程序需要的時間長度對應增長的有多快。
2.處理某個程序所處理某個特定數據的效率不能衡量該程序的好或坏，而是要等規模擴大數倍甚至數百數千萬倍，運行的時間是否和當時小規模的相似，或者有變慢數百倍數千萬倍。
3. 不管數據有多大，程序所處理時間始終花這麽長，這個程序也就是具有O(1)的時間複雜度
4.數據規模變得有多大， 花時間的也跟著變長，找個n個數字的最大值這個程序的時間複雜夫就是O(n), 為綫性級複雜函數。
5.冒泡排序和插入排序等，數據擴大2倍， 時間變慢4倍，時間複雜度就是O(n^2),為平方級別複雜度。
6.窮舉法，所需時間長度成幾何階數上漲，這就是O(a^n)指數級複雜度，甚至O(n!)。
```

#### 關於NP難問題的三個事實
1. 普遍性：與實際相關和NP難問題無處在。
2. 難處理性：在一個被廣汎數學猜想下，任何NP問題可以由任何始終認爲正確且始終運行的算法多項式時間。
3. 不是死刑：NP難題通常可以（但并不總是）在實踐中至少通過足夠的資源投資和算法複雜度。


![Alt text](image-1.png)

#### 常見還原範例：
1. 求整數數組的中值可歸結為 排序數組的問題。 （排序後數組，返回中間元素。）

2. 所有對的最短路徑問題歸結為單源最短路徑問題。 （調用單個源最短路徑算灋每種可能一次輸入圖中起始頂點的選擇。）

3. 最長公共子序列問題簡化為序列比對問題。 （調用序列兩個輸入字串的對齊算法，a每插入一個空位1個罰分，以及一個非常大的罰分對於兩個不同符號的每個不匹配。）

#### 從舊算法中創造出新算法
1. 還原將MergeSort 算法轉化爲O(n log n)  時間中值查找算法，更廣汎地說，任何T(n)時間的排序算法轉化爲O(T(n)) - time 中的值查找算法。其中n是數組長度。
2. 縮減將單源最短路徑問題的任何T(m,n)時間算法轉化為 O(n-T(n))。時間算法轉化爲O(n-T(m,n))時間的全對最短路徑問題算法，其中m和n分別表示邊的數量和頂點數量。
3. 將序列排序問題的T(m,n)時間算法轉化為最長公共子序列問題的O(t(m,n))時間算法，其中，m和n表示兩個輸入字符串的長度。

## NP 問題背景

NP完備性由1971年引入。約翰·霍普克羅夫特 (John Hopcroft) 認為，NP 完全問題是否可以在多項式時間內解決的問題應該推遲到以後，因為無論如何都沒有人能夠正式證明他們的主張。並取得了與會各方的共識。。就是所謂的“是否P=NP問題”。 

### P = NP？
```
* 如果有些演算法用DTM來做計算所需時間是polynomial time, 那這些演算法或問題被稱爲P問題。
* 另外有些演算法使用NTM來做計算所需時間是polynomial time, 那這類問題被稱爲NP問題。
```
NP問題還有另一個數學上等價的判斷方法，，從驗證解的難度來界定，如果用DTM來驗證一組解是否正確只需要 polynomial time，那這個問題就是一個NP問題，剛剛子集合加總問題，我們要驗證解是否正確很簡單也很快速，我們只要把解的數字加總起來看是不是為0就可以了，所以子集合加總問題是一個NP問題，但因為這個問題的時間複雜度為 O(N×(2N))，所以它不是一個Ｐ問題。

```
Definition of NP problem:
NTM 可在 polynomial time 内解決問題 ≡ 問題的解有辦法在DTM polynomial time下被驗證
```

兩個定義爲什麽會被連接起來？因爲NTM有無窮多個分支可以使用，那我就讓每個分支去窮舉每種可能的解，再驗證每個分支的解是否正確，驗證過程只需要 polynomial time, 所以自然在NTM只需要polynomial time 就看可以將這個問題解解完 ，因此是等價的。

### NP-Complete problem
美國Stephen A. Cook提出了Cook-Levin理論，這個數學理論指出任何一個NP裡面的問題都可以在 polynomial time 內，使用DTM，將之化約成「一個布林方程式是否存在解」的問題，這個被化約的問題又稱為布爾可滿足性問題（SAT），我們稱SAT問題為NP-Complete問題。
```
滿足以下兩點 都稱之爲NP-Complete
1. [問題]本身是一個NP問題
2. 所有的NP問題都可以用DTM在 polynomial time 内化約成這個[問題]。
```
這個概念假設我證明了SAT 是P問題 就等於今天我隨便拿到一個NP 問題就可以在polynomial time 內把問題轉換成SAT，然後再用 polynomial time 把SAT解掉，所以所有的NP問題都只是P問題了，也就是P=NP，因此NP-Complete問題就是解決 P=NP 的關鍵，如果可以證明NP-Complete問題為P問題，就可以間接證明P=NP。

### NP-Complete(NPC)
需滿足兩個條件：
1. 這是一個 NP 問題
2. 所有屬於NP問題都可以規約成它
換而言之，只要解決這個問題，所有的NP問題都解決了。

*可歸約*: 將一個問題轉化成另一個問題，使用第二個問題的解來解第一個問題，這種思想類似於數學證明中， 如果一個證明，很難從原題切入，此時根據原命題與其逆否命題是等價的，將得到及其簡便的解法或是結題切入點。
歸化具有傳遞性。也就是説問題A可以歸化成問題B，B可以歸化成C，A一定可以歸化C。

### NP-Hard問題
NP-Hard 問題即滿足NPC問題的第二個條件但不一定滿足第一個條件，因此NP-Hard問題要比NPC問題更範圍廣闊，NP-Hard問題不一定是NP問題。

#### 結序
左圖為假設 P≠NP被證明的情形，NP-Hard有兩個部分，一個部分它同時是個NP問題，另外一部分則不是，所謂的NP問題就是可以用NTM在 polynomial time內給解掉的問題，另外其解的驗證必定能用DTM在 polynomial time內完成，兩種定義是等價的，有一部分的NP問題是屬於P問題，這些問題大部分都是易解的，有另外一部分的NP問題為NP-Complete問題。

右圖是假設P＝NP被證明的情形，此時NP-Complete問題已經被證明為P問題，利用NP-Complete問題的特性，我們可以化約所有NP問題為NP-Complete問題，在把這個NP-Complete問題用 polynomial time 解掉，所以P=NP=NP-Complete。


![Alt text](image.png)
```
```
# 程式碼由chatgpt 全權編寫 我讀懂了
# 簡單 示例 由chatgpt生成
def is_satisfiable(variables, clauses, assignment):
    for clause in clauses:
        satisfied = False
        for literal in clause:
            variable, is_negated = abs(literal), literal < 0
            if (variable in assignment and assignment[variable] == (not is_negated)):
                satisfied = True
                break
        if not satisfied:
            return False
    return True

def verify_3sat():
    # 3SAT问题的例子
    variables = {1, 2, 3}
    clauses = [(1, 2, -3), (-1, 2, 3), (-1, -2, -3)]

    # 假设的解
    assignment = {1: True, 2: False, 3: True}

    # 验证解是否正确
    if is_satisfiable(variables, clauses, assignment):
        print("解是正确的")
    else:
        print("解是错误的")

if __name__ == "__main__":
    verify_3sat()
```
