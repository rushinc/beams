import numpy as np
import scipy as sp
from numpy import linalg as la
from bisect import bisect
import time
import beams as bm
from beams import *

class Cell:
    def __init__(self, period, N, layers):
        self._p = to_vec2(period)
        self._N = to_vec2(N, dtype=int)
        self._layers = list(layers)
        self.reset()

    def reset(self):
        """Forces reset. Stored solutions are discarded."""
        self._freq = None
        self._k = None
        self._u = None
        self._B = None
        self._C = None
        self._reset = True

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, period):
        p = to_vec2(period)
        if p != self.p:
            self._p = p
            if not self._reset: self.reset()

    @property
    def N(self):
        """Number of Fourier modes. Vector2d of positive odd ints."""
        return self._N

    @N.setter
    def N(self, new_N):
        N =  to_vec2(new_N)
        if N != self.N:
            self._N = N
            if not self._reset: self.reset()
            # Ensure N is odd.
            # TODO: Allow even N.
            if not self._N.x % 2:
                self._N.x += 1 
            if not self._N.y % 2:
                self._N.y += 1 

    @property
    def N_t(self):
        return self.N.x * self.N.y

    @property
    def L(self):
        return len(self.layers)

    @property
    def freq(self):
        """Frequency of the last computation."""
        return self._freq

    @property
    def k(self):
        """Wavevector of the last computation."""
        return self._k

    @property
    def B(self):
        """Matrix form of the linear system."""
        return self._B

    @property
    def C(self):
        """The eigenmode amplitude coefficients for each layer."""
        return self._C

    @property
    def u(self):
        """Electric field polarization vector of the last computation."""
        return self._u

    @property
    def layers(self):
        """List of layers in the cell arranged in order of increasing z-coordinate."""
        return self._layers

    @layers.setter
    def layers(self, new_layers):
        self._layers = list(new_layers)
        if not self._reset: self.reset()

    def build(self, freq, k):
        """
        Builds the linear system for a given frequency and wavevector.

        This function computes the eigenvalues of all the layers in the cell
        and builds a 4 * N * (L + 1) square matrix. No return value but the
        constructed matrix can be accessed via `cell.B`.

        Parameters
        ==========
        freq : double
            The frequency at which to compute the layer eigenvalues.
        k : Vector2d
            The wavevector along the plane of the layers.
        """

        L = self.L
        N_t = self.N_t
        B = np.zeros((4 * N_t * (L + 1), 4 * N_t * (L + 1)), dtype=complex)
        Z = np.zeros((2 * N_t, 2 * N_t), dtype=complex)
        for i, l in enumerate(self.layers):
            l.compute_eigs(freq, k, self.p, self.N)
            b_l = np.block([[l.U, l.U @ l.X], [l.V, -l.V @ l.X],
                [-l.U @ l.X, -l.U], [-l.V @ l.X, l.V]])
            B[4 * i * N_t:4 * (i + 2) * N_t,
                    4 * i * N_t:4 * (i + 1) * N_t] = b_l
            if i == 0:
                B[:4 * N_t, 4 * N_t * L:] = np.block([[-l.U, Z], [l.V, Z]])
            if i == L - 1:
                B[-4 * N_t:, 4 * N_t * L:] = np.block([[Z, l.U], [Z, l.V]])
        self._B = B
        self._freq = freq
        self._k = k

    def linsolve(self, freq, k, b):
        """
        Solves the linear system arising from the boundary conditions.

        Often invoked after `cell.build()`, this is where the linear system is
        actually solved. Since `cell.B` can be constructed independently of the
        source fields the to methods can be accessed separately. No values are
        returned but solutions can be accessed via `cell.C`.

        Parameters
        ==========
        freq : double
            The frequency at which to compute the layer eigenvalues.
        k : Vector2d
            The wavevector along the plane of the layers.
        b : np.ndarray
            A flattened 4*N*(L+1) array of the field eigenmode amplitudes.
        """

        L = self.L
        N_t = self.N_t

        if self.freq != freq or self.k != k:
            self.build(freq, k)

        C = la.solve(self.B, b)
        self._C = np.reshape(C, (4 * N_t, L + 1), order='F')
        self._C[2 * N_t:4 * N_t, L - 1] = np.zeros(2 * N_t, dtype=complex)

    def angles_to_k(self, freq, angles):
        """
        Converts the electric field orientation to an in-plane wavevector.

        Parameter
        =========
        freq : double
            The frequency of the source
        angles : Vector3d
            The orientation of the electric field along the three angular
            degrees of freedom.
        """

        theta = angles.x
        phi = angles.y
        psi = angles.z
        kx = self.layers[0].material.get('n', freq) * np.outer(np.sin(theta),
                np.cos(phi))
        ky = self.layers[0].material.get('n', freq) * np.outer(np.sin(theta),
                np.sin(phi))
        ks = Vector2d(kx, ky).flatten()
        return ks

    def angles_to_u(self, angles):
        """
        Converts the electric field orientation to in-plane amplitude components.
        
        Parameters
        ==========
        angles : Vector3d
            The orientation of the electric field along the three angular
            degrees of freedom.
        """
        
        theta = angles.x
        phi = angles.y
        psi = angles.z
        ux = np.zeros((theta.size * phi.size, psi.size))
        uy = np.zeros((theta.size * phi.size, psi.size))
        for (i, p) in enumerate(psi.flat):
            px = np.cos(p) * np.outer(np.cos(theta),
                    np.cos(phi)) - np.sin(p) * np.sin(phi)
            py = np.cos(p) * np.outer(np.cos(theta),
                    np.sin(phi)) + np.sin(p) * np.cos(phi)
            ux[:, i] = px.flatten()
            uy[:, i] = py.flatten()
        return Vector2d(ux, uy)

    def __excitation(self, u, mode_id=0, layer_id=0):
        """
        Computes the eigenmode field components of the source field.

        Converts the the source electric field in-plane amplitude components to
        the corresponding eigenmode amplitudes and returns an array to be used
        in `cell.linsolve`.

        Parameters
        ==========
        u : Vector2d
            The in-plane components of the source electric field vector.
        mode_id : int, optional
            The eigenmode index of the source as per the eigenmodes of
            `cell.layers[layer].
        layer_id : int, optional
            The index of the layer in which to apply the source. Defaults to 0.
        """

        L = self.L
        N_t = self.N_t
        l = layer_id
        exc = np.zeros((2 * N_t, 1))
        exc[N_t // 2 + mode_id] = u.x
        exc[N_t + N_t // 2 + mode_id] = u.y
        ui = np.vstack((exc, self.layers[l].V @ exc))
        self._u = u
        return np.vstack((np.zeros((4 * N_t * l, 1)), ui,
            np.zeros((4 * N_t * (L - l), 1))))

    def diffraction_orders(self, freq=None, angles=None):
        L = self.L
        N_t = self.N_t
        if angles is not None:
            u = self.angles_to_u(angles)
            k = self.angles_to_k(freq, angles)
        else:
            if self.u is None:
                raise AttributeError('Cell does not contain any solutions. ' +\
                        'R() missing 2 required positional arguments: ' +\
                        '\'freq\', and \'angles\'')
                return
            u = self.u
            k = self.k

        if (freq is None or k is None) and self.C is None:
            raise AttributeError('Cell does not contain any solutions. ' +\
                    'R() missing 2 required positional arguments: ' +\
                    '\'freq\', and \'angles\'')
            return
        elif (freq is not None and freq != self.freq) or\
                (k is not None and k != self.k):
            self.build(freq, k)
            b = self.__excitation(u)
            self.linsolve(freq, k, b)

        g_r = self.layers[0].gamma[:N_t]
        g_t = self.layers[-1].gamma[:N_t]
        rx = self.C[:N_t, L]
        ry = self.C[N_t:2 * N_t, L]
        rz = self.layers[0].K.diag().dot(Vector2d(rx,
            ry)) / g_r
        tx = self.C[2 * N_t:3 * N_t, L]
        ty = self.C[3 * N_t:, L]
        tz = -self.layers[-1].K.diag().dot(Vector2d(tx,
            ty)) / g_t
        return (Vector3d(rx, ry, rz), Vector3d(tx, ty, tz))

    def R_T(self, freq=None, angles=None):
        N_t = self.N_t
        (r, t) = self.diffraction_orders(freq, angles)
        g_r = self.layers[0].gamma[:N_t]
        g_t = self.layers[-1].gamma[:N_t]
        ri = np.real(g_r / g_r[N_t // 2]) @ r.intensity()
        ti = np.real(g_t / g_r[N_t // 2]) @ t.intensity()
        return (ri, ti)

    def fields(self, freq, angles, pts):
        L = self.L
        N_t = self.N_t
        theta = angles.x
        phi = angles.y
        psi = angles.z

        u = self.angles_to_u(angles)
        k = self.angles_to_k(freq, angles)

        self.build(freq, k)
        b = self.__excitation(u)
        self.linsolve(freq, k, b)

        e_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        e_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        e_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_x = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_y = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        h_z = np.zeros((pts.x.size, pts.y.size, pts.z.size), dtype=complex)
        E = Vector3d(e_x, e_y, e_z)
        H = Vector3d(h_x, h_y, h_z)

        zi = np.cumsum([l.h for l in self.layers])
        for (k, z) in enumerate(pts.z.flatten()):
            li = bisect(zi[:-1], z)
            l = self.layers[li]
            if li > 0: zz = z - zi[li - 1]
            else: zz = z
            c = self.C[:, [li]]
            amp = Vector2d(np.hstack((c[:N_t], c[2 * N_t:3 * N_t])),
                    np.hstack((c[N_t:2 * N_t], c[3 * N_t:])))
            pts_n = Vector3d(pts.x, pts.y, zz)
            (E[:, :, [k]], H[:, :, [k]]) = l.get_fields(pts_n, amp)

        return (E, H)

    def spectrum(self, freqs, angles):
        L = self.L
        N_t = self.N_t
        theta = angles.x
        phi = angles.y
        psi = angles.z

        R = np.empty((len(freqs), theta.size, phi.size, psi.size))
        T = np.empty((len(freqs), theta.size, phi.size, psi.size))

        u = self.angles_to_u(angles)

        for (i, f) in enumerate(freqs):
            ks = self.angles_to_k(f, angles)
            R_k = np.zeros((theta.size * phi.size, psi.size))
            T_k = np.zeros((theta.size * phi.size, psi.size))
            for (j, k) in enumerate(ks):
                self.build(f, k)
                for l in range(psi.size):
                    b = self.__excitation(u[j, l])

                    self.linsolve(f, k, b)
                    (r, t) = self.diffraction_orders()
                    (ri, ti) = self.R_T()
                    prog = 100 * (l + 1 + j * psi.size +\
                            i * theta.size * phi.size * psi.size) / R.size
                    print('Progress: ' + str(round(prog, 2)) + '%: ' +\
                            'R = ' + str(ri) + ', T = ' + str(ti), end='\r')

                    R_k[j, l] = ri
                    T_k[j, l] = ti
            R[i, :, :, :] = np.reshape(R_k, (theta.size, phi.size, psi.size))
            T[i, :, :, :] = np.reshape(T_k, (theta.size, phi.size, psi.size))

        return (np.squeeze(R), np.squeeze(T))

    def convergence(self, N_max, **kwargs):
        N_i = Vector2d(xy=np.arange(1, N_max + 1, 2, dtype=int)) 
        R = np.zeros(N_i.shape)
        T = np.zeros(N_i.shape)
        DT = np.zeros(N_i.shape)
        for i, n in enumerate(N_i):
            self.N = n
            t0 = time.time()
            (R[i], T[i]) = self.R_T(**kwargs)
            print('N = ' + str(n) + ': R = ' + str(round(R[i], 5)) +
                    ', T = ' + str(round(T[i], 5)))
            self.reset()
            t1 = time.time()
            dt = t1 - t0
            DT[i] = dt
            print(str(round(DT[i], 3)) + 's taken per iteration.')
        return ((R, T), DT)

