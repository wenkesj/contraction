// Package contraction implements tensor contraction for matrix multiplication specific tasks.
package contraction

import (
  "sync"
)

type contract struct {
  dims, da, db             int
  ashape, bshape, cshape []int
  a, b, c                []float64
}

func (c *contract) index(indices []int, x int) (int, int) {
  aindices := append([]int(nil), indices...)
  bindices := append([]int(nil), indices...)
  aindices[c.db], bindices[c.da] = x, x
  aindex, bindex := aindices[0], bindices[0]
  for i, k := c.dims - 2, 1; i >= 0; i, k = i - 1, k + 1 {
    aindex = (aindex * c.ashape[k] + aindices[k])
    bindex = (bindex * c.bshape[k] + bindices[k])
  }
  return aindex, bindex
}

func (c *contract) kernel(ri, rf, n int, indices []int) {
  var thread, div, prod, i, x, aindex, bindex int
  var sum float64
  for thread = ri; thread < rf; thread++ {
    i = c.dims - 1
    prod = c.cshape[i]
    indices[i] = thread % prod
    for i = c.dims - 2; i >= 0; i-- {
      div = c.cshape[i]
      indices[i] = (thread / prod) % div
      prod *= div
    }

    sum = 0
    for x = 0; x < n; x++ {
      aindex, bindex = c.index(indices, x)
      sum += c.a[aindex] * c.b[bindex]
    }
    c.c[thread] = sum
  }
}

// Contract computes a tensor contraction along the leading dimensions of a and b and stores the
// result into c.
//
// `blockSize` is a parameter to choose the level of parallelism for each kernel. The thread
// scatters the array with the given `blockSize`, so it is suggested that `blockSize < len(c)` in
// order to take advantage of the scatter method of efficient contraction.
func Contract(blockSize int, a, b, c []float64, ashape, bshape, cshape []int) {
  size := len(c)
  cdims := len(cshape)
  if len(ashape) != cdims || len(bshape) != cdims {
    panic("`ashape`, `bshape` and `cshape` must have the same rank.")
  }
  if blockSize > size {
    panic("`blockSize` must be <= `len(c)`")
  }

  var blocks, ri, rf int
  var wg *sync.WaitGroup
  con := &contract{
    dims: cdims,
    da: cdims - 2, db: cdims - 1,
    ashape: ashape,
    bshape: bshape,
    cshape: cshape,
    a: a, b: b, c: c,
  }

  wg = new(sync.WaitGroup)
  blocks = size / blockSize
  n := con.ashape[con.dims - 1]

  for ri, rf = 0, blocks; rf <= size; ri, rf = ri + blocks, rf + blocks {
    wg.Add(1)
    go func (ri, rf int, wg *sync.WaitGroup)  {
      defer wg.Done()
      con.kernel(ri, rf, n, make([]int, con.dims))
    }(ri, rf, wg)
  }

  wg.Wait()

  if rf < size {
    con.kernel(ri, size, n, make([]int, con.dims))
  }
}
