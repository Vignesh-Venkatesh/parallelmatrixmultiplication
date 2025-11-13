package main

import (
	"fmt"
	"math/rand"
)

// custom type Matrix
type Matrix [][]int

// generating random matrix with P rows and Q columns
func GenerateRandomMatrix(P int, Q int, num int) Matrix {
    M:=make(Matrix, P) // creatung a slice of P rows
    for i:=range M {
        M[i]=make([]int, Q) // for each row, create a slice of Q columns
        for j:=0; j<Q; j++ {
            // generating a random number between -num and num
            // rand.Intn(21) returns 0<=n<21
            M[i][j]=rand.Intn(2*num+1)-num
        }
    }
    return M
}

func ClassicMatrixMultiply(A, B Matrix) (Matrix, error) {
    // P is rows of A
    P:=len(A)
    if P==0{
        return nil, fmt.Errorf(("matrix A is empty"))
    } 

    // Q is cols of A and rows of B
    Q_A:=len(A[0])
    if Q_A==0 {
        return  nil, fmt.Errorf(("matrix A has zero columns"))
    }
    Q_B:=len(B)
    if Q_B==0 {
        return nil, fmt.Errorf("dimensions do not match. A columns (%d) != B rows (%d)", Q_A, Q_B)
    }

    // R is cols of B
    R:=len(B[0])
    if R==0 {
        return nil, fmt.Errorf(("matrix B has zero columns"))
    }

    // initializing the result matrix C
    C:=make(Matrix, P)
    for i:=range C {
        C[i] = make([]int, R)
    }   

    // classical approach is of time complexity O(P*Q*R) which is basically three for loops
    for i:=0; i<P; i++ { // rows of C (P)
        for j:=0; j<R; j++{ // columns of C (R)
            sum:=0
            for k:=0; k<Q_A; k++{ // inner dimension (Q)
                sum+=A[i][k]*B[k][j]
            }
            C[i][j]=sum
        }
    }

    return C, nil
}

func main() {
    fmt.Println("Starting Matrix Multiplication")
    A := GenerateRandomMatrix(2,2, 20)
    B := GenerateRandomMatrix(2,2, 20)

    fmt.Println("matrix A:", A)
    fmt.Println("matrix B:", B)

    C, error := ClassicMatrixMultiply(A, B)
    if error != nil {
        fmt.Println("Error:", error)
        return
    }
    fmt.Println("matrix C (AxB):", C)
}