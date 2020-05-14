import numpy as np
import eigsc
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

matrixDims = [12, 48, 120, 240, 360, 480]

for matrixDim in matrixDims:
    matrix = np.zeros(matrixDim*matrixDim).reshape(matrixDim,matrixDim)

    if rank == 0: 
        matrix = np.random.rand(matrixDim,matrixDim)
        
        before = time()
        vals2, vecs2 = eigsc.eigsc(matrix)
        after = time()
        scipyTime = after - before

    before = time()
    comm.Bcast(matrix,root=0)

    vals, vecs = eigsc.eigsc_slepc(matrix)

    gatherVecs = None
    if rank == 0:
        gatherVecs = np.empty([matrixDim,matrixDim], dtype=complex)

    comm.Gather(vecs,gatherVecs,root = 0)
    after = time()
    slepcTime = after - before

    if rank == 0:
        valNorm = np.linalg.norm(np.sort_complex(vals2) - np.sort_complex(vals))
        #print("Eigenvalue Norm Difference: " + str(valNorm))
        #print("Scipy Time: " + str(scipyTime) + ", SLEPc Time: " + str(slepcTime))
        print(str(matrixDim) + " " + str(scipyTime) + " " + str(slepcTime))

