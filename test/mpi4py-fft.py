import sys
sys.path.append('/home/rushin/Documents/Python')

import beams as bm
import numpy as np
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

"""
N = np.array([128, 128], dtype=int)
fft = PFFT(MPI.COMM_WORLD, N, axes=(0, 1), dtype=np.float, grid=(-1,))
u = newDistArray(fft, False)
u[:] = np.random.random(u.shape).astype(u.dtype)
u_hat = fft.forward(u, normalize=True) # Note that normalize=True is default and can be omitted
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj)
assert np.allclose(uj, u)
print(MPI.COMM_WORLD.Get_rank(), u.shape)

"""

np.random.seed(0)
test = np.random.rand(4, 4)
if not rank:
    print(test)
N = np.array([4, 4], dtype=int)
fft = PFFT(comm, N, axes=(1, 0), dtype=np.float)

u = newDistArray(fft, False)
u[:] = test[:, int(np.ceil(rank * 4 / size)):int(np.ceil((rank + 1) * 4 / size))]
comm.Barrier()
print(comm.Get_rank(), "\n", u)
u_hat = fft.forward(u, normalize=False)
uj = np.zeros_like(u)
uj = fft.backward(u_hat, uj, normalize=True)
assert np.allclose(uj, u)

print(comm.Get_rank(), "\n", u_hat)
if not rank:
    npf = np.fft.fftshift(np.fft.fft2(test))
    print(npf)

recvbuf = None
if not rank:
    recvbuf = np.empty([4, 4], dtype=complex)
comm.Gather(u_hat, recvbuf, root=0)

print(comm.Get_rank(), "\n", recvbuf)

