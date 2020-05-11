import numpy as np
import eigsc
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

if rank == 0: 
    matrix = np.random.rand(500,500)


    before = time()
    vals2, vecs2 = eigsc.eigsc(matrix)
    after = time()
    scipyTime = after - before
    print(scipyTime)
    sortVals2 = np.sort_complex(vals2)

    
    before = time()
    vals, vecs = eigsc.eigsc_slepc(matrix)
    after = time()
    petscTime = after - before
    print(petscTime)
    sortVals = np.sort_complex(vals)

    valNorm = np.linalg.norm(np.sort_complex(vals2) - np.sort_complex(vals))
    vecNorm = np.linalg.norm(vecs2 - vecs)

    print(str(valNorm) + " " + str(vecNorm))
    print(str(petscTime) + ", " + str(scipyTime))

#for i in range(100):
#    print(str(i) + ": " + str(vals[i]) + ", " + str(vals2[i]))

print(rank)
