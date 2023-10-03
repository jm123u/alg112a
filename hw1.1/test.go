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

