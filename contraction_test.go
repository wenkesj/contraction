package contraction

import (
  "fmt"
  "math/rand"
  "sync"
  "testing"
  "time"

  "gonum.org/v1/gonum/blas"
  "gonum.org/v1/gonum/blas/blas64"
)

func randData(shape []int) []float64 {
  var i int
  var size int = 1
  for _, i = range shape {
    size *= i
  }
  data := make([]float64, size)
  for i = 0; i < size; i++ {
    data[i] = rand.Float64()
  }
  return data
}

func generateShapes(dims int, ashape []int) ([]int, []int, []int) {
  bshape := append([]int(nil), ashape...)
  bshape[dims - 2], bshape[dims - 1] = ashape[dims - 1], ashape[dims - 2]
  cshape := make([]int, dims)
  cshape[dims - 2], cshape[dims - 1] = ashape[dims - 2], bshape[dims - 1]
  if dims > 2 {
    for i := 0; i < dims - 2; i++ {
      cshape[i] = ashape[i]
    }
  }
  return ashape, bshape, cshape
}

func TestMatrixMultiplication(t *testing.T) {
  ashape, bshape, cshape := generateShapes(2, []int{100, 100})
  adata, bdata, cdata := randData(ashape), randData(bshape), randData(cshape)
  aBlas := blas64.General{
  	Rows: ashape[0], Cols: ashape[1],
  	Stride: ashape[1],
  	Data: adata,
  }
  bBlas := blas64.General{
  	Rows: bshape[0], Cols: bshape[1],
  	Stride: bshape[1],
  	Data: bdata,
  }
  cBlas := blas64.General{
  	Rows: cshape[0], Cols: cshape[1],
  	Stride: cshape[1],
  	Data: append([]float64(nil), cdata...),
  }

  wg := new(sync.WaitGroup)
  wg.Add(2)
  go func ()  {
    defer wg.Done()
    ti := time.Now()
    Contract(100, adata, bdata, cdata, ashape, bshape, cshape)
    te := time.Now()
    elapsed := te.Sub(ti)
    fmt.Println("Contract", elapsed)
  }()

  go func ()  {
    defer wg.Done()
    ti := time.Now()
    blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aBlas, bBlas, 0, cBlas)
    te := time.Now()
    elapsed := te.Sub(ti)
    fmt.Println("Gemm", elapsed)
  }()
  wg.Wait()

  for i := 0; i < len(cdata); i++ {
    if cdata[i] != cBlas.Data[i] {
      t.Errorf("c != cBlas at index %d: got %f != %f", i, cdata[i], cBlas.Data[i])
    }
  }
}
