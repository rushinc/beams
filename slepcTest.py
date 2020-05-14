import numpy as np
import eigsc
from time import time
from mpi4py import MPI


matrixDim = 400
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
localRows = matrixDim // size 


#print(rank)

matrix = np.zeros(matrixDim*matrixDim).reshape(matrixDim,matrixDim)

if rank == 0: 
    matrix = np.random.rand(matrixDim,matrixDim)
    
    before = time()
    vals2, vecs2 = eigsc.eigsc(matrix)
    after = time()
    scipyTime = after - before
    #print("Scipy Time: " + str(scipyTime))
    #sortVals2 = np.sort_complex(vals2)

    #print(vals2)
    #print(vecs2)
    
#if rank == 0:
    #matrix_split = np.array_split(matrix, size, axis = 0)
    #split_sizes = []
    #for i in range(0,len(split),1):
     #   split_sizes = np.append(split_sizes, len(split[i]))



before = time()
#print("Rank " + str(rank) + " has before matrix " + str(matrix))
comm.Bcast(matrix,root=0)
#print("Rank " + str(rank) + " has after matrix " + str(matrix))    

vals, vecs = eigsc.eigsc_slepc(matrix)

#print(slepcTime)
#sortVals = np.sort_complex(vals)

#print("Rank " + str(rank) + " has eigenvectors " + str(vecs))


gatherVecs = None
if rank == 0:
    gatherVecs = np.empty([matrixDim,matrixDim], dtype=complex)

comm.Gather(vecs,gatherVecs,root = 0)
after = time()
slepcTime = after - before
#if rank == 0:
#    print(vals)
#    print(gatherVecs)



if rank == 0:
    valNorm = np.linalg.norm(np.sort_complex(vals2) - np.sort_complex(vals))
    #vecNorm = np.linalg.norm(vecs2 - vecs)
    print("Eigenvalue Norm Difference: " + str(valNorm))
    #print(str(valNorm) + " " + str(vecNorm))
    print("Scipy Time: " + str(scipyTime) + ", SLEPc Time: " + str(slepcTime))

#for i in range(100):
#    print(str(i) + ": " + str(vals[i]) + ", " + str(vals2[i]))

#print(rank)
