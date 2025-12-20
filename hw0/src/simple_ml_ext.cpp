#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t iterations = (m + batch - 1)/batch;
    for (size_t i =0;i<iterations;++i)
    {
      const float* x = X + i * batch * n;
      const unsigned char* yy = y + i * batch;
      if (i == iterations - 1)
        batch = m - batch * (iterations -1);
      float *Z = new float[batch * k];

      for (size_t a = 0;a< batch;++a) {
        float za = 0;
        for (size_t b = 0; b< k;++b) {
          float zab = 0;
          for (size_t c = 0;c < n;++c)
            zab += x[a*n+c] * theta[c*k+b];
          Z[a*k+b] = exp(zab);
          za += exp(zab);
        }
        for (size_t b =0;b< k;++b) Z[a*k+b] /= za;
        Z[a*k+yy[a]] -= 1;
      }

      for (size_t a=0; a< n;++a)
        for (size_t b =0;b< k;++b) {
          float gab = 0;
          for (size_t c =0;c< batch;++c) {
            gab += x[c*n + a] * Z[c*k+b];
          }
          theta[a*k+b] -= lr * gab/batch;
        }
      delete[] Z;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
