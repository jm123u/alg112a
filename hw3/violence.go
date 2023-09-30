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
