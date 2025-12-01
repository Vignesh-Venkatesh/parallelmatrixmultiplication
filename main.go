package main

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"time"
)

// custom type Matrix
type Matrix [][]int

// reference to the submatrix
type MatrixView struct {
	data Matrix // referencing the actual matrix
	row_start int // position of the row's start of the submatrix
	col_start int // position of the col's start of the submatrix
	size int // dimension of the submatrix
}

// function to generate a random matrix
func GenerateRandomMatrix(rows int, cols int, r int) Matrix {
	matrix := make(Matrix, rows)

	for row:=0; row<rows; row++ {
		matrix[row] = make([]int, cols)
		for col:=0; col<cols; col++ {
			matrix[row][col] = rand.IntN(2*r+1)-r
		}
	}

	return matrix
}

func ClassicMatrixMultiplication(A Matrix, B Matrix) Matrix {
	A_rows:=len(A)
	A_cols:=len(A[0])

	B_rows:=len(B)
	B_cols:=len(B[0])

	// checking if dimension mismatch exists

	// case 1: checking if A is a square matrix
	if A_rows!=A_cols{
		panic("A is not a square matrix")
	}
	// case 2: checking if B is a square matrix
	if B_rows!=B_cols{
		panic("B is not a square matrix")
	}
	// case 3: checking if col of A and row of B match
	if A_cols!=B_rows{
		panic("number of columns of A and number of rows of B do not match")
	}

	// creating a resulting matrix
	C := make(Matrix, A_rows)
	for i:=0; i<A_rows; i++ {
		C[i]=make([]int,B_cols)
	}

	// matrix multiplication - 3 for loops
    // i = row of A
    // j = column of A (or row of B)
    // k = column of B

	for i := 0; i < A_rows; i++ {
        for j := 0; j < A_cols; j++ {
            for k := 0; k < B_cols; k++ {
                C[i][k] += A[i][j] * B[j][k]
            }
        }
    }

	return C
}

func ParallelClassicMatrixMultiplication(A Matrix, B Matrix) Matrix {
	A_rows:=len(A)
	A_cols:=len(A[0])

	B_rows:=len(B)
	B_cols:=len(B[0])

	// checking if dimension mismatch exists

	// case 1: checking if A is a square matrix
	if A_rows!=A_cols{
		panic("A is not a square matrix")
	}
	// case 2: checking if B is a square matrix
	if B_rows!=B_cols{
		panic("B is not a square matrix")
	}
	// case 3: checking if col of A and row of B match
	if A_cols!=B_rows{
		panic("number of columns of A and number of rows of B do not match")
	}

	// creating a resulting matrix
	C := make(Matrix, A_rows)
	for i:=0; i<A_rows; i++ {
		C[i]=make([]int,B_cols)
	}

	var wg sync.WaitGroup



	// matrix multiplication - 3 for loops
    // i = row of A
    // j = column of A (or row of B)
    // k = column of B

	for i:=0; i<A_rows; i++ {
		wg.Add(1)
		go func(row int) {
			defer wg.Done()
			for j:=0; j<A_cols; j++ {
				for k:=0; k<B_cols; k++ {
					C[row][k]+=A[row][j]*B[j][k]
				}
			}
		}(i)
	}

	wg.Wait()

	return C
}

// getting element at position (i,j) within the submatrix view
func (view MatrixView) GetEl(i int, j int) int {
	return view.data[view.row_start+i][view.col_start+j]
}

// setting elmeent at position (i,j) within the submatrix view
func (view MatrixView) SetEl(i int, j int, val int) {
	view.data[view.row_start+i][view.col_start+j]=val
}

// creating a subview for individual quadrants
func (view MatrixView) SubView(quadrant int) MatrixView {
	half:=view.size/2

	switch quadrant{
	// top left
	case 0:
		return MatrixView{view.data, view.row_start, view.col_start, half}
	// top right
	case 1:
		return MatrixView{view.data, view.row_start, view.col_start+half, half}
	// bottom left
	case 2:
		return MatrixView{view.data, view.row_start+half, view.col_start, half}
	// bottom left
	case 3:
		return MatrixView{view.data, view.row_start+half, view.col_start+half, half}
	}

	return view
}

// helper function to add two matrix views and store result in dest
func AddInPlace(dest, src1, src2 MatrixView) {
	for i:=0; i<dest.size; i++ {
		for j:=0; j<dest.size; j++ {
			dest.SetEl(i,j,src1.GetEl(i,j)+src2.GetEl(i,j))
		}
	}
}

// divide and conquer
func DivideAndConquerRecursive(A MatrixView, B MatrixView, C MatrixView) {
	n:=A.size

	// base case of 1x1 matrix
	if n==1 {
		C.SetEl(0,0,A.GetEl(0,0)*B.GetEl(0,0))
		return
	}

	// creating subviews to track indices
	A11,A12,A21,A22:=A.SubView(0),A.SubView(1),A.SubView(2),A.SubView(3)
	B11,B12,B21,B22:=B.SubView(0),B.SubView(1),B.SubView(2),B.SubView(3)
	C11,C12,C21,C22:=C.SubView(0),C.SubView(1),C.SubView(2),C.SubView(3)

	// allocating temporary matrices for intermediate steps
	// (we need these because we do: C11=A11*B11+A12*B21)
	half:=n/2
	temp1:=make(Matrix,half)
	temp2:=make(Matrix,half)
	
	for i:=0; i<half; i++ {
		temp1[i]=make([]int,half)
		temp2[i]=make([]int,half)
	}

	temp1View:=MatrixView{temp1,0,0,half}
	temp2View:=MatrixView{temp2,0,0,half}

	// C11=A11*B11+A12*B21
	DivideAndConquerRecursive(A11,B11,temp1View)
	DivideAndConquerRecursive(A12,B21,temp2View)
	AddInPlace(C11,temp1View,temp2View)

	// C12=A11*B12+A12*B22
	DivideAndConquerRecursive(A11,B12,temp1View)
	DivideAndConquerRecursive(A12,B22,temp2View)
	AddInPlace(C12,temp1View,temp2View)

	// C21=A21*B11+A22*B21
	DivideAndConquerRecursive(A21,B11,temp1View)
	DivideAndConquerRecursive(A22,B21,temp2View)
	AddInPlace(C21,temp1View,temp2View)

	// C22=A21*B12+A22*B22
	DivideAndConquerRecursive(A21,B12,temp1View)
	DivideAndConquerRecursive(A22,B22,temp2View)
	AddInPlace(C22,temp1View,temp2View)
}

// main divide and conquer function
func DivideAndConquerMatrixMultiply(A Matrix, B Matrix) Matrix {
	n:=len(A)

	// allocating result matrix (only one allocation for the entire computation)
	C:=make(Matrix, n)
	for i:=0; i<n; i++ {
		C[i]=make([]int,n)
	}

	// creating views for the entire matrices
	A_view:=MatrixView{A,0,0,n}
	B_view:=MatrixView{B,0,0,n}
	C_view:=MatrixView{C,0,0,n}

	// running the recursive algo
	DivideAndConquerRecursive(A_view,B_view,C_view)

	return C
}

// divide and conquer - parallel
func ParallelDivideAndConquerRecursive(A MatrixView, B MatrixView, C MatrixView, depth int, maxDepth int) {
	n:=A.size

	// base case of 1x1 matrix
	if n==1 {
		C.SetEl(0,0,A.GetEl(0,0)*B.GetEl(0,0))
		return
	}

	// creating subviews to track indices
	A11,A12,A21,A22:=A.SubView(0),A.SubView(1),A.SubView(2),A.SubView(3)
	B11,B12,B21,B22:=B.SubView(0),B.SubView(1),B.SubView(2),B.SubView(3)
	C11,C12,C21,C22:=C.SubView(0),C.SubView(1),C.SubView(2),C.SubView(3)

	// allocating temporary matrices for intermediate steps
	// (we need these because we do: C11=A11*B11+A12*B21)
	half:=n/2
	temp1:=make(Matrix,half)
	temp2:=make(Matrix,half)
	
	for i:=0; i<half; i++ {
		temp1[i]=make([]int,half)
		temp2[i]=make([]int,half)
	}

	temp1View:=MatrixView{temp1,0,0,half}
	temp2View:=MatrixView{temp2,0,0,half}

	// if we are at the max depth, we continue sequentially
	if depth >= maxDepth {

		// C11=A11*B11+A12*B21
		DivideAndConquerRecursive(A11,B11,temp1View)
		DivideAndConquerRecursive(A12,B21,temp2View)
		AddInPlace(C11,temp1View,temp2View)

		// C12=A11*B12+A12*B22
		DivideAndConquerRecursive(A11,B12,temp1View)
		DivideAndConquerRecursive(A12,B22,temp2View)
		AddInPlace(C12,temp1View,temp2View)

		// C21=A21*B11+A22*B21
		DivideAndConquerRecursive(A21,B11,temp1View)
		DivideAndConquerRecursive(A22,B21,temp2View)
		AddInPlace(C21,temp1View,temp2View)

		// C22=A21*B12+A22*B22
		DivideAndConquerRecursive(A21,B12,temp1View)
		DivideAndConquerRecursive(A22,B22,temp2View)
		AddInPlace(C22,temp1View,temp2View)

		return
	}

	// parallelizing the four quadrant computations
	var wg sync.WaitGroup
	wg.Add(4)

	// C11=A11*B11+A12*B21
	go func() {
		defer wg.Done()
		temp1_c11:=make(Matrix,half)
		temp2_c11:=make(Matrix,half)
		for i:=0; i<half; i++ {
			temp1_c11[i]=make([]int,half)
			temp2_c11[i]=make([]int,half)
		}
		temp1View_c11:=MatrixView{temp1_c11,0,0,half}
		temp2View_c11:=MatrixView{temp2_c11,0,0,half}

		ParallelDivideAndConquerRecursive(A11,B11,temp1View_c11,depth+1,maxDepth)
		ParallelDivideAndConquerRecursive(A12,B21,temp2View_c11,depth+1,maxDepth)
		AddInPlace(C11,temp1View_c11,temp2View_c11)
	}()

	// C12=A11*B12+A12*B22
	go func() {
		defer wg.Done()
		temp1_c12:=make(Matrix,half)
		temp2_c12:=make(Matrix,half)
		for i:=0; i<half; i++ {
			temp1_c12[i]=make([]int,half)
			temp2_c12[i]=make([]int,half)
		}
		temp1View_c12:=MatrixView{temp1_c12,0,0,half}
		temp2View_c12:=MatrixView{temp2_c12,0,0,half}

		ParallelDivideAndConquerRecursive(A11,B12,temp1View_c12,depth+1,maxDepth)
		ParallelDivideAndConquerRecursive(A12,B22,temp2View_c12,depth+1,maxDepth)
		AddInPlace(C12,temp1View_c12,temp2View_c12)
	}()

	// C21=A21*B11+A22*B21
	go func() {
		defer wg.Done()
		temp1_c21:=make(Matrix,half)
		temp2_c21:=make(Matrix,half)
		for i:=0; i<half; i++ {
			temp1_c21[i]=make([]int,half)
			temp2_c21[i]=make([]int,half)
		}
		temp1View_c21:=MatrixView{temp1_c21,0,0,half}
		temp2View_c21:=MatrixView{temp2_c21,0,0,half}

		ParallelDivideAndConquerRecursive(A21,B11,temp1View_c21,depth+1,maxDepth)
		ParallelDivideAndConquerRecursive(A22,B21,temp2View_c21,depth+1,maxDepth)
		AddInPlace(C21,temp1View_c21,temp2View_c21)
	}()

	// C22=A21*B12+A22*B22
	go func() {
		defer wg.Done()
		temp1_c22:=make(Matrix,half)
		temp2_c22:=make(Matrix,half)
		for i:=0; i<half; i++ {
			temp1_c22[i]=make([]int,half)
			temp2_c22[i]=make([]int,half)
		}
		temp1View_c22:=MatrixView{temp1_c22,0,0,half}
		temp2View_c22:=MatrixView{temp2_c22,0,0,half}

		ParallelDivideAndConquerRecursive(A21,B12,temp1View_c22,depth+1,maxDepth)
		ParallelDivideAndConquerRecursive(A22,B22,temp2View_c22,depth+1,maxDepth)
		AddInPlace(C22,temp1View_c22,temp2View_c22)
	}()

	wg.Wait()
}

// main divide and conquer function - parallel
func ParallelDivideAndConquerMatrixMultiply(A Matrix, B Matrix) Matrix {
	n:=len(A)

	// allocating result matrix (only one allocation for the entire computation)
	C:=make(Matrix, n)
	for i:=0; i<n; i++ {
		C[i]=make([]int,n)
	}

	// creating views for the entire matrices
	A_view:=MatrixView{A,0,0,n}
	B_view:=MatrixView{B,0,0,n}
	C_view:=MatrixView{C,0,0,n}

	// using depth threshold to only parallelize first few levels so that we avoid excessive goroutines
	maxDepth:=3

	// running the parallel recursive algo
	ParallelDivideAndConquerRecursive(A_view,B_view,C_view,0,maxDepth)

	return C
}

func BenchmarkAlgo(A Matrix, B Matrix, algo func(Matrix, Matrix) Matrix, runs int) time.Duration {
	var total_time time.Duration

	for i:=0; i<runs; i++{
		start:=time.Now()
		algo(A,B)
		total_time+=time.Since(start)
	}

	return total_time/time.Duration(runs)

}

func main(){
	fmt.Printf("MATRIX MULTIPLICATION\n\n")

	num_rows:=512 // for setting dimension of the matrix
	num_runs:=5 // for benchmarking

	fmt.Printf("Dimension of matrix: %v x %v\n",num_rows, num_rows)
	fmt.Printf("Number of runs: %v\n\n",num_runs)

	A_rows, A_cols, A_range_limit := num_rows,num_rows,10
	A:=GenerateRandomMatrix(A_rows, A_cols, A_range_limit)
	// fmt.Println() 
	// fmt.Println("matrix A:",A)

	B_rows, B_cols, B_range_limit := num_rows,num_rows,10
	B:=GenerateRandomMatrix(B_rows, B_cols, B_range_limit)
	// fmt.Println("matrix B:",B)
	
	// start:=time.Now()
	// // fmt.Println() 
	// // C_classic:=ClassicMatrixMultiplication(A, B)
	// // fmt.Println("matrix C (classic):",C_classic)
	// ClassicMatrixMultiplication(A,B)
	// time_classic:=time.Since(start)
	// fmt.Printf("Classic Matrix Multiplication took: %v\n",time_classic)
	
	// start=time.Now()
	// // fmt.Println() 
	// // C_dc:=DivideAndConquerMatrixMultiply(A, B)
	// // fmt.Println("matrix C (divide and conquer):",C_dc)
	// DivideAndConquerMatrixMultiply(A,B)
	// time_dc:=time.Since(start)
	// fmt.Printf("Divide and Conquer Matrix Multiplication took: %v\n",time_dc)

	//-------------------------------------------------------------------
	// benchmarking classic algo
	fmt.Printf("Running Classic Matrix Multiplication\n")
	avg_time_classic:=BenchmarkAlgo(A,B,ClassicMatrixMultiplication,num_runs)
	fmt.Printf("Average time (Classic): %v\n\n",avg_time_classic)
	
	// benchmarking classic algo (parallel)
	fmt.Printf("Running Parallel Classic Matrix Multiplication\n")
	avg_time_classic_parallel:=BenchmarkAlgo(A,B,ParallelClassicMatrixMultiplication,num_runs)
	fmt.Printf("Average time (Classic - Parallel): %v\n\n",avg_time_classic_parallel)

	speedup_classic:=float64(avg_time_classic)/float64(avg_time_classic_parallel)
	fmt.Printf("Speedup (classic): %.2fx\n\n", speedup_classic)
	//-------------------------------------------------------------------
	
	//-------------------------------------------------------------------
	// benchmarking Divide and Conquer Matrix Multiplication
	fmt.Printf("Running Divide and Conquer Matrix Multiplication\n")
	avg_time_DC:=BenchmarkAlgo(A,B,DivideAndConquerMatrixMultiply,num_runs)
	fmt.Printf("Average time (Divide & Conquer): %v\n",avg_time_DC)

	// benchmarking Divide and Conquer Matrix Multiplication (parallel)
	fmt.Printf("Running Parallel Divide and Conquer Matrix Multiplication\n")
	avg_time_DC_parallel:=BenchmarkAlgo(A,B,ParallelDivideAndConquerMatrixMultiply,num_runs)
	fmt.Printf("Average time (Divide & Conquer - Parallel): %v\n",avg_time_DC_parallel)

	speedup_DC:=float64(avg_time_DC)/float64(avg_time_DC_parallel)
	fmt.Printf("Speedup (divide & conquer): %.2fx\n\n", speedup_DC)
	//-------------------------------------------------------------------
	
	//-------------------------------------------------------------------
	// // speedup comparison
	// if avg_time_classic>avg_time_DC {
	// 	speedup:=float64(avg_time_classic)/float64(avg_time_DC)
	// 	fmt.Printf("Divide & Conquer is %.2fx faster\n",speedup)
	// } else {
	// 	speedup:=float64(avg_time_DC)/float64(avg_time_classic)
	// 	fmt.Printf("Classic is %.2fx faster\n",speedup)
	// }
	//-------------------------------------------------------------------
	
	//-------------------------------------------------------------------
	// overall comparison
	fmt.Printf("OVERALL COMPARISON\n")
	fmt.Printf("Classic (Sequential): %v\n", avg_time_classic)
	fmt.Printf("Classic (Parallel): %v (%.2fx faster)\n", avg_time_classic_parallel, speedup_classic)
	fmt.Printf("D&C (Sequential): %v\n", avg_time_DC)
	fmt.Printf("D&C (Parallel): %v (%.2fx faster)\n", avg_time_DC_parallel, speedup_DC)
	//-------------------------------------------------------------------


}