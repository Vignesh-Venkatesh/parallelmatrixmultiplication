package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// custom type Matrix
type Matrix [][]int


// returns the smallest power of two >= n
func NextPowerOfTwo(n int) int {
	if n<=0 {
		return 1
	}
	// math.Ceil(math.Log2(float64(n))) finds the exponent needed
	// we then use math.Pow to find 2 raised to that exponent
	return int(math.Pow(2, math.Ceil(math.Log2(float64(n)))))
}

// takes a matrix M and pads it with zeros to size NxN
func PadMatrix(M Matrix, N int) Matrix {
	P:=len(M)
	Q:=len(M[0])

	Padded:=make(Matrix, N)
	for i:=0; i<N; i++ {
		Padded[i]=make([]int, N)
		if i<P {
			// copying the existing row segment and leaving the rest as zeros
			copy(Padded[i][:Q], M[i])
		}
	}
	return Padded
}

// extracts the original P x R result from the padded N x N matrix.
func TrimMatrix(Padded Matrix, P, R int) Matrix {
	C:=make(Matrix, P)
	for i:=0; i<P; i++ {
		C[i]=make([]int, R)
		// copying only the first R columns
		copy(C[i], Padded[i][:R])
	}
	return C
}


// generating random matrix with P rows and Q columns
func GenerateRandomMatrix(P int, Q int, num int) Matrix {
	M := make(Matrix, P) // creatung a slice of P rows
	for i := range M {
		M[i] = make([]int, Q) // for each row, create a slice of Q columns
		for j := 0; j < Q; j++ {
			// generating a random number between -num and num
			M[i][j] = rand.Intn(2*num+1) - num
		}
	}
	return M
}

// classic matrix multiplication using three for loops
func ClassicMatrixMultiply(A, B Matrix) (Matrix, error) {
	// P is rows of A
	P:=len(A)
	if P==0 {
		return nil, fmt.Errorf("matrix A is empty")
	}

	// Q is cols of A and rows of B
	Q_A:=len(A[0])
	if Q_A==0 {
		return nil, fmt.Errorf("matrix A has zero columns")
	}
	Q_B:=len(B) // rows of B
	if Q_B!=Q_A {
		return nil, fmt.Errorf("dimension mismatch: A columns (%d) != B rows (%d)", Q_A, Q_B)
	}

	// R is cols of B
	R:=len(B[0])
	if R==0 {
		return nil, fmt.Errorf("matrix B has zero columns")
	}

	// initializing the result matrix C
	C:=make(Matrix, P)
	for i:=range C {
		C[i]=make([]int, R)
	}

	// classical approach is of time complexity O(P*Q*R) which is basically three for loops
	for i:=0; i<P; i++ { // rows of C (P)
		for j:=0; j<R; j++ { // columns of C (R)
			sum:=0
			for k:=0; k<Q_A; k++ { // inner dimension (Q)
				sum+=A[i][k]*B[k][j]
			}
			C[i][j]=sum
		}
	}

	return C, nil
}


// helper function - adding two matrices
func Add(A, B Matrix) Matrix {
	n:=len(A)
	C:=make(Matrix, n)
	for i:=0; i<n; i++ {
		C[i]=make([]int, n)
		for j:=0; j<n; j++ {
			C[i][j]=A[i][j]+B[i][j]
		}
	}
	return C
}

// helper function - getting submatrices, extracting an n/2 x n/2 block from M starting at (row,col)
func SubMatrix(M Matrix, row, col, size int) Matrix {
	S:=make(Matrix, size)
	for i:=0; i<size; i++ {
		S[i]=make([]int, size)
		copy(S[i], M[row+i][col:col+size])
	}
	return S
}

// helper function - combining a matrix C from its four n/2 x n/2 sub blocks
func Combine(C11, C12, C21, C22 Matrix, n int) Matrix {
	C:=make(Matrix, n)
	half:=n/2
	for i:=0; i<n; i++ {
		C[i]= make([]int, n)
		if i<half {
			// top half, we copy C11 then C12
			copy(C[i][:half], C11[i])
			copy(C[i][half:], C12[i])
		} else {
			// bottom half, we copy C21 then C22
			copy(C[i][:half], C21[i-half])
			copy(C[i][half:], C22[i-half])
		}
	}
	return C
}

// divide and conquer, recursive O(N^3) function
// called only with NxN matrices where N is a power of 2
func DivideAndConquerRecursive(A, B Matrix) Matrix {
	n:=len(A)

	// base case = 1x1 matrix
	if n==1 {
		result:=make(Matrix, 1)
		result[0]=[]int{A[0][0]*B[0][0]}
		return result
	}

	mid:=n/2

	// divide step, getting the eight N/2 x N/2 submatrices
	A11:=SubMatrix(A, 0, 0, mid)
	A12:=SubMatrix(A, 0, mid, mid)
	A21:=SubMatrix(A, mid, 0, mid)
	A22:=SubMatrix(A, mid, mid, mid)

	B11:=SubMatrix(B, 0, 0, mid)
	B12:=SubMatrix(B, 0, mid, mid)
	B21:=SubMatrix(B, mid, 0, mid)
	B22:=SubMatrix(B, mid, mid, mid)

	// conquer step, 8 multiplications
	P1:=DivideAndConquerRecursive(A11, B11)
	P2:=DivideAndConquerRecursive(A12, B21)
	P3:=DivideAndConquerRecursive(A11, B12)
	P4:=DivideAndConquerRecursive(A12, B22)
	P5:=DivideAndConquerRecursive(A21, B11)
	P6:=DivideAndConquerRecursive(A22, B21)
	P7:=DivideAndConquerRecursive(A21, B12)
	P8:=DivideAndConquerRecursive(A22, B22)

	// combine step, 4 additions to get Cij blocks
	C11:=Add(P1, P2)
	C12:=Add(P3, P4)
	C21:=Add(P5, P6)
	C22:=Add(P7, P8)

	// reassembling C
	return Combine(C11, C12, C21, C22, n)
}

// function for handling padding and matrix multiplication
func DivideAndConquerMatrixMultiply(A, B Matrix) (Matrix, error) {
	P:=len(A)
	Q:=len(A[0])
	R:=len(B[0])

	if Q!=len(B) {
		return nil, fmt.Errorf("dimensions do no tmatch: A columns (%d) != B rows (%d)", Q, len(B))
	}

	// determining padding size: finding the next power of two for the largest dimension
	N:=NextPowerOfTwo(int(math.Max(math.Max(float64(P), float64(Q)), float64(R))))

	// padding matrices to NxN
	A_padded:=PadMatrix(A, N)
	B_padded:=PadMatrix(B, N)

	// running divide and conquer algo
	C_padded:=DivideAndConquerRecursive(A_padded, B_padded)

	// trimming result to extract the PxR submatrix
	C:=TrimMatrix(C_padded, P, R)

	return C, nil
}


func main() {

	fmt.Println("Matrix Multiplication")

	// test 1: non power of two matrix (e.g., 3x3) to verify padding logic 
	P, Q, R := 3, 3, 3 // testing 3x3 * 3x3 = 3x3, padded size will be 4

	A:=GenerateRandomMatrix(P, Q, 10)
	B:=GenerateRandomMatrix(Q, R, 10)

	// classic matrix multiplication algo result
	C_classic, err := ClassicMatrixMultiply(A, B)
	if err != nil {
		fmt.Println("Classic Error:", err)
		return
	}

	// divide and conquer algo result (with padding and trimming)
	C_dc, err := DivideAndConquerMatrixMultiply(A, B)
	if err != nil {
		fmt.Println("D&C Error:", err)
		return
	}

	fmt.Println("Matrix A:", A)
	fmt.Println("Matrix B:", B)
	fmt.Println("Classic C:", C_classic)
	fmt.Println("D&C C:", C_dc)


	// performance check 
	N_perf:=512 
	fmt.Printf("\nPerformance Check: %d x %d \n", N_perf, N_perf)
	A_perf:=GenerateRandomMatrix(N_perf, N_perf, 10)
	B_perf:=GenerateRandomMatrix(N_perf, N_perf, 10)

	// classic algo timing
	start_classic:=time.Now()
	ClassicMatrixMultiply(A_perf, B_perf)
	duration_classic:=time.Since(start_classic)

	// divide and conquer algo timing
	start_dc:=time.Now()
	DivideAndConquerMatrixMultiply(A_perf, B_perf)
	duration_dc:=time.Since(start_dc)

	fmt.Printf("Classic Algo Sequential Time: %v\n", duration_classic)
	fmt.Printf("Divide and Conquer Algo Sequential Time: %v\n", duration_dc)
}