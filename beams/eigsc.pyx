cimport cython
import numpy as np
cimport numpy as np
import scipy.linalg as linalg
import slepc4py, petsc4py
from petsc4py import PETSc
from slepc4py import SLEPc


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

#cpdef (DTYPE_t [:], DTYPE_t[:,:]) eigs(matrixA, matrixB):
cpdef eigsc(matrix):
    vals, vecs = linalg.eig(matrix)
    return vals, vecs

cpdef eigsc_slepc(np.ndarray[DTYPE_t, ndim=2, mode="c"] matrix):
    comm = PETSc.COMM_WORLD
    size = comm.getSize()
    rank = comm.getRank()

    m = matrix.shape[0]
    #print("here")
    #print(str(size) + ", " + str(rank) + ", " + str(m))
    
    
    localLowRow = int(rank*m/size)
    localHighRow = int((rank+1)*m/size)

    #A = PETSc.Mat().createDense([m,m],array=&matrix[0,0])
    A = PETSc.Mat().createDense([m,m], array=matrix[localLowRow:localHighRow,:],comm=comm)
    A.assemble()
    E = SLEPc.EPS().create(comm=comm)
    E.setOperators(A)
    E.setDimensions(m,PETSc.DECIDE)
    E.solve()


    nconv = E.getConverged()
    vr, vi = A.getVecs();
    eigenvalues = np.zeros(m, dtype = complex)
    eigenvectors = np.zeros([m//size,m], dtype = complex)
    for i in range(nconv):
        eigenvalues[i] = E.getEigenpair(i, vr, vi)
        #print(vr.getArray().shape)
        #print("Rank " + str(rank) + " has " + str(i) + " eigenvalue " + str(eigenvalues[i]) + " with eigenvector " + str(vr.getArray()) + " " + str(vi.getArray())
        eigenvectors[:,i] = vr.getArray() + vi.getArray()*1j
        #print("Rank " + str(rank) + " has " + str(i) + " eigenvector " + str(eigenvectors))

    #print("Rank " + str(rank) + " has vals " + str(eigenvalues) + " and vecs " + str(eigenvectors) +  ".")


    #print("Number of converged eigenpairs: " + str(nconv) +".")
    #print(eigenvalues)
    return eigenvalues, eigenvectors 
    #return eigs_actualC(&matrix[0,0],
