# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:00:31 2022

@author: ZR
"""
#%% Easiest cuda code to get used to cuda.
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
mod = SourceModule("""
 __global__ void doublify(float *a)
 {
 int idx = threadIdx.x + threadIdx.y*4;
 a[idx] *= 2;
 }
 """)
func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))
a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print (a_doubled)
print (a)

#%% a relatively complicate function to compare CPU and GPU calculation speed.
# The more complex the calculation is, the quicker GPU can be.
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer
from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
  {
    return;
  }
  float temp_a = a[i];
  float temp_b = b[i];
  a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
  // a[i] = a[i] + b[i];
}
""")

func = mod.get_function("func")   

def test(N):
    # N = 1024 * 1024 * 90   # float: 4M = 1024 * 1024

    print("N = %d" % N)

    N = np.int32(N)
    
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)   
    # copy a to aa
    aa = np.empty_like(a)
    aa[:] = a
    # GPU run
    nTheads = 256
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    start = timer()
    func(
            drv.InOut(a), drv.In(b), N,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
    run_time = timer() - start  
    print("gpu run time %f seconds " % run_time)    
    # cpu run
    start = timer()
    aa = (aa * 10 + 2 ) * ((b + 2) * 10 - 5 ) * 5
    run_time = timer() - start  

    print("cpu run time %f seconds " % run_time)  

    # check result
    r = a - aa
    print( min(r), max(r) )

def main():
  for n in range(1, 10):
    N = 1024 * 1024 * (n * 10)
    print("------------%d---------------" % n)
    test(N)

if __name__ == '__main__':
    main()