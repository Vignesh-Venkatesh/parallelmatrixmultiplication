package main

import (
	"fmt"
	"math/rand/v2"
	"runtime"
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


// ------------------------------------------------------------------
// Benchmarking function

func BenchmarkAlgo(A Matrix,B Matrix,algo func(Matrix, Matrix) Matrix,warmups int,runs int,) time.Duration {

	fmt.Printf("\n")
    // warming up to ensure stable performance
    for i:=0; i<warmups; i++ {
		algo(A,B)
    }
	fmt.Printf("WARMUP DONE\n\n")
	
    // benchmarking runs
    total:=time.Duration(0)
    fmt.Println("Runs:")
	
    for i:=0; i<runs; i++ {
		start:=time.Now()
        algo(A,B)
        elapsed:=time.Since(start)

        fmt.Printf("  Run %-2d: %v\n",i+1,elapsed)
        total+=elapsed
    }

    avg:=total/time.Duration(runs)
    fmt.Printf("  Average: %v\n\n", avg)

    return avg
}


// ------------------------------------------------------------------
// Correctness testing

func MatrixEqual(A, B Matrix) bool {
	for i:=0; i<len(A); i++ {
		for j:=0; j<len(A[i]); j++ {
			if A[i][j]!=B[i][j] {
				return false
			}
		}
	}

	return true
}

func CorrectnessTesting() {

	// ---------------------------------------------------------
	// case 1: multiplying two non zero matrices
	fmt.Printf("\nCase 1: multiplying two non zero matrices\n")
	fmt.Printf("===========================================\n")
	A:=Matrix{
		{1,2},
		{3,4},
	}
	B:=Matrix{
		{5,6},
		{7,8},
	}
	expected_result:=Matrix{
		{19,22},
		{43,50},
	}

	result_classic:=ClassicMatrixMultiplication(A,B);
	result_classic_parallel:=ParallelClassicMatrixMultiplication(A,B)
	result_dandc:=DivideAndConquerMatrixMultiply(A,B);
	result_dandc_parallel:=ParallelDivideAndConquerMatrixMultiply(A,B);

	// checking if classic (sequential) matched
	if MatrixEqual(result_classic,expected_result) {
		fmt.Println("Classic Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Sequential) Test Failed!")
	}
	// checking if classic (parallel matched)
	if MatrixEqual(result_classic_parallel,expected_result) {
		fmt.Println("Classic Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Parallel) Test Failed!")
	}
	// checking if d and c (sequential matched)
	if MatrixEqual(result_dandc,expected_result) {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Failed!")
	}
	// checking if d and c (parallel matched)
	if MatrixEqual(result_dandc_parallel,expected_result) {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Failed!")
	}

	// ---------------------------------------------------------
	// case 2: multiplying by the identity matrix
	fmt.Printf("\nCase 2: multiplying by the identity matrix\n")
	fmt.Printf("===========================================\n")
	A=Matrix{
		{1,2},
		{3,4},
	}
	B=Matrix{
		{1,0},
		{0,1},
	}
	expected_result=Matrix{
		{1,2},
		{3,4},
	}

	result_classic=ClassicMatrixMultiplication(A,B);
	result_classic_parallel=ParallelClassicMatrixMultiplication(A,B)
	result_dandc=DivideAndConquerMatrixMultiply(A,B);
	result_dandc_parallel=ParallelDivideAndConquerMatrixMultiply(A,B);

	// checking if classic (sequential) matched
	if MatrixEqual(result_classic,expected_result) {
		fmt.Println("Classic Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Sequential) Test Failed!")
	}
	// checking if classic (parallel matched)
	if MatrixEqual(result_classic_parallel,expected_result) {
		fmt.Println("Classic Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Parallel) Test Failed!")
	}
	// checking if d and c (sequential matched)
	if MatrixEqual(result_dandc,expected_result) {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Failed!")
	}
	// checking if d and c (parallel matched)
	if MatrixEqual(result_dandc_parallel,expected_result) {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Failed!")
	}

	// ---------------------------------------------------------
	// case 3: multiplying by the zero matrix
	fmt.Printf("\nCase 3: multiplying by the zero matrix\n")
	fmt.Printf("===========================================\n")
	A=Matrix{
		{1,2},
		{3,4},
	}
	B=Matrix{
		{0,0},
		{0,0},
	}
	expected_result=Matrix{
		{0,0},
		{0,0},
	}

	result_classic=ClassicMatrixMultiplication(A,B);
	result_classic_parallel=ParallelClassicMatrixMultiplication(A,B)
	result_dandc=DivideAndConquerMatrixMultiply(A,B);
	result_dandc_parallel=ParallelDivideAndConquerMatrixMultiply(A,B);

	// checking if classic (sequential) matched
	if MatrixEqual(result_classic,expected_result) {
		fmt.Println("Classic Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Sequential) Test Failed!")
	}
	// checking if classic (parallel matched)
	if MatrixEqual(result_classic_parallel,expected_result) {
		fmt.Println("Classic Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Parallel) Test Failed!")
	}
	// checking if d and c (sequential matched)
	if MatrixEqual(result_dandc,expected_result) {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Failed!")
	}
	// checking if d and c (parallel matched)
	if MatrixEqual(result_dandc_parallel,expected_result) {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Failed!")
	}

	// ---------------------------------------------------------
	// case 4: multiplying with negative numbers
	fmt.Printf("\nCase 4: multiplying with negative numbers\n")
	fmt.Printf("===========================================\n")
	A=Matrix{
		{1,-2},
		{-3,4},
	}
	B=Matrix{
		{-5,6},
		{7,-8},
	}
	expected_result=Matrix{
		{-19,22},
		{43,-50},
	}

	result_classic=ClassicMatrixMultiplication(A,B);
	result_classic_parallel=ParallelClassicMatrixMultiplication(A,B)
	result_dandc=DivideAndConquerMatrixMultiply(A,B);
	result_dandc_parallel=ParallelDivideAndConquerMatrixMultiply(A,B);

	// checking if classic (sequential) matched
	if MatrixEqual(result_classic,expected_result) {
		fmt.Println("Classic Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Sequential) Test Failed!")
	}
	// checking if classic (parallel matched)
	if MatrixEqual(result_classic_parallel,expected_result) {
		fmt.Println("Classic Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Parallel) Test Failed!")
	}
	// checking if d and c (sequential matched)
	if MatrixEqual(result_dandc,expected_result) {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Failed!")
	}
	// checking if d and c (parallel matched)
	if MatrixEqual(result_dandc_parallel,expected_result) {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Failed!")
	}

	// ---------------------------------------------------------
	// case 5: multiplying with all ones
	fmt.Printf("\nCase 5: multiplying with all ones\n")
	fmt.Printf("===========================================\n")
	A=Matrix{
		{1,1},
		{1,1},
	}
	B=Matrix{
		{1,1},
		{1,1},
	}
	expected_result=Matrix{
		{2,2},
		{2,2},
	}

	result_classic=ClassicMatrixMultiplication(A,B);
	result_classic_parallel=ParallelClassicMatrixMultiplication(A,B)
	result_dandc=DivideAndConquerMatrixMultiply(A,B);
	result_dandc_parallel=ParallelDivideAndConquerMatrixMultiply(A,B);

	// checking if classic (sequential) matched
	if MatrixEqual(result_classic,expected_result) {
		fmt.Println("Classic Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Sequential) Test Failed!")
	}
	// checking if classic (parallel matched)
	if MatrixEqual(result_classic_parallel,expected_result) {
		fmt.Println("Classic Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Classic Algo (Parallel) Test Failed!")
	}
	// checking if d and c (sequential matched)
	if MatrixEqual(result_dandc,expected_result) {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Sequential) Test Failed!")
	}
	// checking if d and c (parallel matched)
	if MatrixEqual(result_dandc_parallel,expected_result) {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Passed!")
	} else {
		fmt.Println("Divide and Conquer Algo (Parallel) Test Failed!")
	}

	
	fmt.Printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
}





func main() {
    fmt.Printf("MATRIX MULTIPLICATION\n\n")
	
    num_rows:=512
    num_runs:=5
    num_warmups:=5
	
    fmt.Println("Available CPUs:", runtime.NumCPU())
    fmt.Printf("Matrix dimension: %dx%d\n", num_rows, num_rows)
    fmt.Printf("Runs per algorithm: %d\n\n", num_runs)
	
    A:=GenerateRandomMatrix(num_rows,num_rows,10)
    B:=GenerateRandomMatrix(num_rows,num_rows,10)

	CorrectnessTesting()

    // --------------------------------------------------------------
    // classic algo (sequential)

    fmt.Println("Classic Algorithm (Sequential)")
    avgClassic:=BenchmarkAlgo(A,B,ClassicMatrixMultiplication,num_warmups,num_runs)
	
    // --------------------------------------------------------------
    // classic algo (parallel)
    // --------------------------------------------------------------

    fmt.Println("Classic Algorithm (Parallel)")
	
    procCounts:=[]int{1,2,4,8,16,30,60}
    classicParallelAverages:=make([]time.Duration,len(procCounts))

    for i,p:=range procCounts {
        fmt.Printf("Using %d procs\n", p)
		fmt.Printf("=================================\n")
        runtime.GOMAXPROCS(p)
		
        avg:=BenchmarkAlgo(A,B,ParallelClassicMatrixMultiplication,num_warmups,num_runs)
        classicParallelAverages[i]=avg
    }

    // --------------------------------------------------------------
    // divide and conquer algo (sequential)

    fmt.Println("Divide and Conquer (Sequential)")
    avgDC:=BenchmarkAlgo(A,B,DivideAndConquerMatrixMultiply,num_warmups,num_runs)

    // --------------------------------------------------------------
    // divide and conquer algo (sequential)

    fmt.Println("Divide and Conquer (Parallel)")

    dcParallelAverages:=make([]time.Duration,len(procCounts))

    for i,p:=range procCounts {
		fmt.Printf("Using %d procs\n",p)
		fmt.Printf("=================================\n")
        runtime.GOMAXPROCS(p)
		
        avg:=BenchmarkAlgo(A,B,ParallelDivideAndConquerMatrixMultiply,num_warmups,num_runs)
        dcParallelAverages[i]=avg
    }

    // --------------------------------------------------------------
    // overall summary
	fmt.Println("====================================================")
	fmt.Printf("OVERALL SUMMARY\n")
	
    fmt.Println("\nClassic (Sequential):",avgClassic)
    fmt.Println("\nClassic (Parallel):")
    for i,p:=range procCounts {
        speed:=float64(avgClassic)/float64(classicParallelAverages[i])
        fmt.Printf("  %2d procs -> %v   (speedup: %.2fx)\n",
            p,classicParallelAverages[i],speed)
    }

    fmt.Println("\nDivide and Conquer (Sequential):",avgDC)
    fmt.Println("\nDivide and Conquer (Parallel):")
    for i,p:=range procCounts {
        speed:=float64(avgDC)/float64(dcParallelAverages[i])
        fmt.Printf("  %2d procs -> %v   (speedup: %.2fx)\n",
            p,dcParallelAverages[i],speed)
    }

    fmt.Println("====================================================")
}