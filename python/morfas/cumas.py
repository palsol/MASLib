import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import math
import time as t


class cuMas:
    def __init__(self):
        bulk_comparison_kernel = SourceModule("""
                #include <stdio.h>

                __device__ double distance(double *a, double *b, int dim)
                        {
                            double result = 0;
                            for (int i = 0; i < dim; i++) {
                                result += pow(b[i] - a[i], 2);
                            }
                            return result;
                        }

                __device__ double distanceConst(double *a, int dim)
                        {
                            double result = 0;
                            for (int i = 0; i < dim; i++) {
                                result += pow(a[i] - 1.0, 2);
                            }
                            return result;
                        }


              __device__ double comparison(double *a, int *aForm, double *b, double *bApprox, double *bRestore, int dim)
                        {
                            bApprox[0] = b[aForm[0]];
                            double aprox = 0;
                            for (int i = 1; i < dim; i++) {
                                aprox = b[aForm[i]];
                                int j = 0;

                                for (j = 0; i > j && aprox < bApprox[i - j - 1]; j++) {
                                    aprox = (aprox * (j + 1) + b[aForm[i - j - 1]]) / (j + 2);
                                }

                                bApprox[i] = aprox;
                                for (int k = 1; k <= j; k++) {
                                    bApprox[i - k] = aprox;
                                }
                            }

                            for (int i = 0; i < dim; i++) {
                                bRestore[aForm[i]] = bApprox[i];
                            }

                            for (int i = 0; i < dim; i++) {
                                 bApprox[i] = bRestore[i];
                            }

                            double result = distance(b, bApprox, dim) / distanceConst(b, dim);
                            return result;
                        }

              __global__ void comp(double *A, int *AForm,
                                   double *b, int *bForm, double *BApprox, double *BRestore,
                                   int dim, int n, double *out)
                        {
                            int idx = blockIdx.x * blockDim.x + threadIdx.x;
                            double *a = A + blockIdx.x * dim;
                            double* bApprox = BApprox + blockIdx.x * dim;
                            double* bRestore = BRestore + blockIdx.x * dim;
                            int *aForm = AForm + blockIdx.x * dim;
                            out[blockIdx.x] = comparison(a, aForm, b, bApprox, bRestore, dim);
                            out[blockIdx.x] += comparison(b, bForm, a, bApprox, bRestore, dim);
                            out[blockIdx.x] /= 2.0;
                         }
              """)
        self.__bulk_comparison = bulk_comparison_kernel.get_function('comp')

    def proximity_vectors_to_b(self, vectors, b, vectors_forms, b_form):
        n = vectors.shape[0]
        dim = b.shape[0]
        if dim != vectors.shape[1]:
            print(dim)
            print(vectors.shape[1])
            raise ValueError("chunks shape must be equal!")

        vectors = vectors.astype(np.double)
        vectors_forms = vectors_forms.astype(np.int)
        b = b.astype(np.double)
        b_form = b_form.astype(np.int)
        b_aprox = np.zeros((n, dim), dtype=np.double)
        b_restore = np.zeros((n, dim), dtype=np.double)
        out = np.zeros(n, dtype=np.double)

        self.__bulk_comparison(cuda.In(vectors), cuda.In(vectors_forms),
                               cuda.In(b), cuda.In(b_form), cuda.In(b_aprox), cuda.In(b_restore),
                               np.uint32(dim), np.uint32(n), cuda.InOut(out), block=(1, 1, 1),
                               grid=(n, 1, 1))
        return out
