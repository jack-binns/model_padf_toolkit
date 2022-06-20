import numpy as np
import vol
from scipy.special import spherical_jn
import os
import sys


class Blqq:

    def __init__(self, l, nq):
        self.l = l
        self.nq = nq
        self.data = np.zeros((nq, nq))


class blqqarray:

    def __init__(self, nl, nq=-1, qmax=-1, rmax=-1, tablenzero=1000, tablelmax=100, tablepath='foo'):

        self.nl = nl
        self.nq = nq
        self.qmax = qmax
        self.rmax = rmax
        self.l = []

        #
        # parameters to read a lookup table of spherical bessel zeros
        #
        self.tablenzero = tablenzero
        self.tablelmax = tablelmax
        self.tablepath = tablepath
        self.jnuzeros = []
        self.read_sphB_zeros()

        #
        # set up the sphB matrices
        #
        if (qmax > 0) and (rmax > 0):
            self.set_up_blqq_array_sphB()
        elif nq > 0:
            self.set_up_blqq_array()
        else:
            print("Error - blqq array cannot be initialised. Check qmax, rmax or nq values.")
            exit()

    def set_up_blqq_array(self):
        self.l = []
        for i in np.arange(self.nl):
            self.l.append(Blqq(i, self.nq))
        print(f"<set_up_blqq_array> {len(self.l)=}")

    def set_up_blqq_array_sphB(self):

        self.l = []
        print(f"{self.nl=}")
        for i in np.arange(self.nl):
            nq = self.sphB_samp_nmax(i, self.rmax, self.qmax)
            print(f"<blqq.set_up_blqq_array_sphB> {nq=}")
            self.l.append(Blqq(i, nq))

    def sphB_samp_nmax(self, l, rmax, qmax):

        qlim = 2 * np.pi * qmax * rmax
        for i in np.arange(self.tablenzero):
            qln = self.jnuzeros[l, i]
            # print("l i qln qlim", l, i, qln, qlim)
            if qln > qlim:
                out = i - 1
                break
        if out < 0:
            out = 0

        return out

    def read_sphB_zeros(self):
        tablename = os.path.join(self.tablepath,
                                 "sphbzeros_lmax" + str(self.tablelmax) + "_nt" + str(self.tablenzero) + ".npy")
        self.jnuzeros = np.load(tablename)

    #
    # resample from zerom from one order to zeros of another
    #
    def resampling_matrix(self, l, l2, nmax, nmax2, qmax, rmax):

        if (qmax < 0) or (rmax < 0):
            print("Error - resampling matrix calc with negative rmax/qmax vals")
            exit()

        qmax2p = 2 * np.pi * qmax

        mat = np.zeros((nmax2, nmax))

        q2 = np.copy(self.jnuzeros[l2, 1:nmax2 + 1] / rmax)

        q = np.copy(self.jnuzeros[l, 1:nmax + 1])
        jl2 = spherical_jn(l + 1, q)
        q *= 1.0 / rmax
        factor2 = np.sqrt(2 * np.pi) / (jl2 * jl2 * (rmax ** 3))

        sb = spherical_jn(l, q)

        r = np.copy(self.jnuzeros[l, 1:nmax + 1] / qmax2p)
        jl1 = jl2
        factor = np.sqrt(2 * np.pi) / (jl1 * jl1 * (qmax2p ** 3))

        for i in np.arange(nmax2):
            for j in np.arange(nmax):
                sb1 = spherical_jn(l, q[j] * r)
                sb2 = spherical_jn(l, q2[i] * r)
                mat[i, j] = np.sum(sb1 * sb2 * factor * factor2)

        return mat
