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
