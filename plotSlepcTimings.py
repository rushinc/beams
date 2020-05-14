import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':22})
plt.figure(figsize=(8,6))

array = np.loadtxt("slepcTimings.txt")

#plt.figure()
plt.plot(array[:,0], array[:,1], label="Serial")
plt.plot(array[:,0], array[:,2], label="MPI-1")
plt.plot(array[:,0], array[:,3], label="MPI-2")
plt.plot(array[:,0], array[:,4], label="MPI-3")
plt.plot(array[:,0], array[:,5], label="MPI-4")
plt.xlabel("Dimension of Matrix")
plt.ylabel("Time (s)")
plt.yscale("log")
plt.legend()
plt.show()
