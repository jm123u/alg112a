/* 
create： 2023.9.30
with : hw3
use: newtonraphson, bisection, regulafalsi
A simple attempt to learn the Go language.
*/

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