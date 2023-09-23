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