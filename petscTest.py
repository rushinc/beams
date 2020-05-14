import numpy as np
import eigsc
from time import time
from mpi4py import MPI


matrixDim = 100
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
localRows = matrixDim // size 


#print(rank)

matrix = np.zeros(matrixDim*matrixDim).reshape(matrixDim, matrixDim)
vector = np.zeros(matrixDim)

if rank == 0: 
    matrix = np.random.rand(matrixDim,matrixDim)
    vector = np.random.rand(matrixDim)
    print("Random Matrix")
    print(matrix)

    print("Random Vector")
    print(vector)

    before = time()
    vec2 = eigsc.solvec(matrix,vector)
    after = time()
    scipyTime = after - before
    #print("Scipy Time: " + str(scipyTime))
    #sortVals2 = np.sort_complex(vals2)

    #print(vals2)
    print("SCIPY X")
    print(vec2)
    
#if rank == 0:
    #matrix_split = np.array_split(matrix, size, axis = 0)
    #split_sizes = []
    #for i in range(0,len(split),1):
     #   split_sizes = np.append(split_sizes, len(split[i]))



before = time()
#print("Rank " + str(rank) + " has before matrix " + str(matrix))
comm.Bcast(matrix,root=0)
comm.Bcast(vector,root=0)
#print("Rank " + str(rank) + " has after matrix " + str(matrix))    

#print(matrix)
#print(vector)

#print(matrix)


print("Trying to solve.")
#vec1 = eigsc.solvec_petsc()
vec1 = eigsc.solvec_petsc(matrix, vector)
print("Done solving")


print("PETSc X")
print(vec1)

#print(vec1)
#print(slepcTime)
#sortVals = np.sort_complex(vals)

#print("Rank " + str(rank) + " has eigenvectors " + str(vecs))


gatherVec = None
if rank == 0:
    gatherVec = np.empty([matrixDim], dtype=np.float64)

comm.Gather(vec1,gatherVec,root = 0)
after = time()
petscTime = after - before
#if rank == 0:
#    print(vals)
#    print(gatherVec)



if rank == 0:
    valNorm = np.linalg.norm(vec2 - vec1)
    #vecNorm = np.linalg.norm(vecs2 - vecs)
    print("Solve Norm Difference: " + str(valNorm))
    #print(str(valNorm) + " " + str(vecNorm))
    print("Scipy Time: " + str(scipyTime) + ", PETSc Time: " + str(petscTime))

#for i in range(100):
#    print(str(i) + ": " + str(vals[i]) + ", " + str(vals2[i]))

#print(rank)
