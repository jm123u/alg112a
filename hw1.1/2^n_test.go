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
