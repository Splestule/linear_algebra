# Linear Algebra
Implementing concepts from linear algebra in python with minimal library usage in the non-test files, while I'm working my way through the 18.06 MIT Linear Algebra course taught by the one and only, great Gilbert Strang.
Having to bring the things I learn into code helps me understand them more thoroughly.

The project includes a solver for ```Ax = b``` with the full ```A(x_p + x_n) = b``` solution, a least squares solver both by gradient descent and by ```A_transposed * Ax = A_transposed * b```, inversion of matrices, a recursive determinant solver, ```PA = LU``` and other functions.

The functions' correctness is verified in the ```/test_solver.py```.
