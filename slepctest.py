import numpy as np
import eigsc
from time import time
from mpi4py import MPI


matrixDim = 4
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
localRows = matrixDim // size 


print(rank)

matrix = np.zeros(matrixDim*matrixDim).reshape(matrixDim,matrixDim)

if rank == 0: 
    matrix = np.random.rand(matrixDim,matrixDim)

    before = time()
    vals2, vecs2 = eigsc.eigsc(matrix)
    after = time()
    scipyTime = after - before
    print(scipyTime)
    sortVals2 = np.sort_complex(vals2)


#if rank == 0:
    #matrix_split = np.array_split(matrix, size, axis = 0)
    #split_sizes = []
    #for i in range(0,len(split),1):
     #   split_sizes = np.append(split_sizes, len(split[i]))

comm.Bcast(matrix,root=0)
    
    
    
before = time()
vals, vecs = eigsc.eigsc_slepc(matrix)
after = time()
petscTime = after - before
print(petscTime)
sortVals = np.sort_complex(vals)

if rank == 0:
    valNorm = np.linalg.norm(np.sort_complex(vals2) - np.sort_complex(vals))
    vecNorm = np.linalg.norm(vecs2 - vecs)

    print(str(valNorm) + " " + str(vecNorm))
    print(str(petscTime) + ", " + str(scipyTime))

#for i in range(100):
#    print(str(i) + ": " + str(vals[i]) + ", " + str(vals2[i]))

#print(rank)
