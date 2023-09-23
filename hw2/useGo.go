// Writing using backtracking
//Using too large a number may take a long time to calculate. Learning GO for the first time.

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

