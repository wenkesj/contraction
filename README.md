# contraction [![GoDoc](https://godoc.org/github.com/wenkesj/contraction?status.svg)](https://godoc.org/github.com/wenkesj/contraction)

This package implements _multi-threaded_, _multi-dimensional_ Tensor Contraction for the special
case of Matrix Multiplication (along contraction dimensions 1 and 0). Currently, the only supported
data type is `float64`.

Tests are not very strong and need improvement over dimensions above 2.

The speed is not very good for large (& dense) Matrix Multiplications due to the use of single
dimensional storage vectors and preprocessing in order to support n-dimensional Matrix
Multiplication for batches of batches of... (i.e. indexing algorithm needs to be improved or
simply defining methods and using a switch case on the dimensions).
