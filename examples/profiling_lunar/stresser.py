import os
import math
import numpy as np

def cpu_stressor():
    import numpy as np

    while True:
        # 大矩阵 GEMM，多线程 BLAS
        a = np.random.rand(1024, 1024).astype(np.float32)
        b = np.random.rand(1024, 1024).astype(np.float32)

        np.matmul(a, b, out=a)



if __name__ == "__main__":
    cpu_stressor()