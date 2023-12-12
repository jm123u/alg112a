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