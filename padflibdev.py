import numpy as np
import matplotlib.pyplot as plt
import vol
import sys
import blqq
from scipy.special import spherical_jn, legendre
import scipy as sp


#
# TO DO:
#   -  check odd/even parts
#   -  check all the matrix multiplcations
#   -  add timings ans outputs
#   -  outputs after each step
#
#   - write a test script; uses new parameter class and new padf class
#     imports the old results from c-code and checks all the outputs
#
#   - after that all works; continue onto the correlation parts of the code

class padfcl:

    def __init__(self, nl=2, nlmin=0, nr=100, nth=69, nq=100, qmin=0.0, qmax=0.0, rmax=0.0, corrvol=None,
                 wl=0.0, units_flag=0, section=1, theta=0, r=0, r2=0,
                 tablenzero=1000, tablelmax=100, tablepath='foo'):

        self.nl = nl
        self.nlmin = nlmin
        self.nr = nr
        self.nth = nth
        self.nq = nq
        self.qmin = qmin
        self.qmax = qmax
        self.rmax = rmax
        self.corrvol = corrvol
        self.wl = wl
        self.units_flag = units_flag  # not implemented

        # padf plot parameters
        self.section = section
        self.theta = theta
        self.r = r
        self.r2 = r2

        #
        # parameters to read a lookup table of spherical bessel zeros
        #
        self.tablepath = tablepath
        self.tablenzero = tablenzero
        self.tablelmax = tablelmax

        #
        # creates and empty volume object
        #
        dimnames = [0, 0, 0]  # "q","q2","theta"]
        dimlen = [self.nq, self.nq, self.nth]
        dmin = [0, 0, 0]
        dmax = [self.qmax, self.qmax, 2.0 * np.pi]

        self.padf = vol.Vol2(dimnames=dimnames, dimlen=dimlen, dmin=dmin, dmax=dmax)

        print("<padflibdev> qmax, rmax:", self.qmax, self.rmax)
        print("<padflibdev> tablenzero tablelmax tablepath", self.tablenzero, self.tablelmax, self.tablepath)
        self.blqq = blqq.blqqarray(self.nl, qmax=self.qmax, rmax=self.rmax,
                                   tablenzero=self.tablenzero,
                                   tablelmax=self.tablelmax,
                                   tablepath=self.tablepath)
        self.blrr = blqq.blqqarray(self.nl, nq=self.nr,
                                   tablenzero=self.tablenzero,
                                   tablelmax=self.tablelmax,
                                   tablepath=self.tablepath)

    #
    #  The main calculation of the padf from the correlation volume
    #
    def padf_calc(self):
        #
        # STEP 1: calculate the B_l(q,q') matrices sampled at the spherical Bessel zeros
        #
        self.Blqq_calc_fast()

        #
        # STEP 2: transform the B_l(q,q') into B_l(r,r') terms
        #
        self.blrr = blqq.blqqarray(self.nl, nq=self.nr,
                                   tablepath=self.tablepath,
                                   tablenzero=self.tablenzero,
                                   tablelmax=self.tablelmax)
        self.Bla_qr_transform()

        #
        # STEP 3: transform B_l(r,r') into the PADF
        #
        self.padf = self.Blrr_to_padf(self.blrr, (self.nr, self.nr, self.nth))

    #
    # Generates self.blqq matrices from the correlation function
    # Each matrix is sampled at the zeros the corresponding spherical bessel function
    #

    # keep regular sampling

    """
    regular sampling method
    corr-vol --> blqq try calcBlrr()
    
    calcBlrr(going backwards and forward)
    
    check dsBmatrix and dsBmatrix_inv
    """


    def Blqq_calc_fast(self):
        self.blqqtmp = blqq.blqqarray(self.nl, nq=self.blqq.l[0].nq,
                                      tablenzero=self.tablenzero, tablelmax=self.tablelmax,
                                      tablepath=self.tablepath)

        # define a list of matrices to store the Blqq matrices
        rsinvlist = []

        for l in np.arange(self.nl):
            # Check passing over nq into here
            mat = self.blqq.resampling_matrix(l, 0, self.blqq.l[l].nq, self.blqq.l[0].nq, self.qmax, self.rmax)
            print(f"{l=} {mat.shape=}")
            inv = sp.linalg.pinv(mat)  # I MAY NOT HAVE TO INVER THIS, BUT JUST SOLVE STUFF
            rsinvlist.append(inv)
        nq = self.blqq.l[0].nq
        print("DEBUG BLQQ_CALC_FAST nq", nq)
        for iq in np.arange(nq):
            for iq2 in np.arange(nq):
                qln1 = self.blqq.jnuzeros[0, iq + 1] / (2 * np.pi * self.rmax)
                qln2 = self.blqq.jnuzeros[0, iq2 + 1] / (2 * np.pi * self.rmax)

                # CALCULATE AND INVERT FMAT; MAY NOT HAVE TO INVERT JUST SOLVE
                mat = self.fmat(qln1, qln2)
                matinv = sp.linalg.pinv2(mat, cond=0.5)

                # FIND THE NEAREST Q INDEX IN THE CORRELATION DATA
                ic = np.round(self.nq * (qln1 / self.qmax)).astype(np.int)
                jc = np.round(self.nq * (qln2 / self.qmax)).astype(np.int)

                tmp = np.dot(matinv.transpose(), self.corrvol[ic, jc, :])

                for k in np.arange(self.nl // 2):
                    self.blqqtmp.l[2 * k].data[iq, iq2] = tmp[k]

                for k in np.arange(self.nl // 2 - 1):
                    self.blqqtmp.l[2 * k + 1].data[iq, iq2] = 0.0
        # RESAMPLE USING THE RESAMPLING MATRICES...
        for k in np.arange(self.nl // 2):
            tmp = np.dot(rsinvlist[2 * k], self.blqqtmp.l[2 * k].data)
            self.blqq.l[2 * k].data = np.dot(tmp, rsinvlist[2 * k].transpose())

    #
    # calculate F-matrix; for conversion of correlation volume to B_l(q,q') matrices
    #
    def fmat(self, q, q2):

        thq = self.thetaq_calc(q, self.wl)
        thq2 = self.thetaq_calc(q2, self.wl)

        phi = 2 * np.pi * np.arange(self.nth) / self.nth
        arg = np.cos(thq) * np.cos(thq2) + np.sin(thq) * np.sin(thq2) * np.cos(phi)
        arg[arg > 1] = 1.0
        arg[arg < -1] = -1.0

        mat = np.zeros((self.nl, self.nth))
        for l in np.arange(self.nl):
            Pn = legendre(int(2 * l))
            mat[l, :] = Pn(arg)
        return mat

    #
    # Calculate the 2 x scattering angle from the magnitude of the q vector
    #
    # We can make this an accelerated calculation with numba in a utils py
    def thetaq_calc(self, q, wl):
        return (np.pi / 2) - np.arcsin(wl * q / 2)

    #
    # transform blqq into real-space blrr matrices
    #
    def Bla_qr_transform(self):

        for k in np.arange(self.nl // 2):
            l = 2 * k
            mat = self.dsBmatrix(l, self.blqq.l[l].nq)
            tmp = np.dot(mat, self.blqq.l[l].data)
            self.blrr.l[l].data = np.dot(tmp, mat.transpose())

    #
    # Calculate the matrix for the q->r transform
    #
    def dsBmatrix(self, l, nq):
        dsbmat = np.zeros((self.nr, nq))
        r = np.arange(self.nr) * self.rmax / self.nr
        for j in np.arange(nq):
            qln = self.blqq.jnuzeros[l, j + 1]
            arg = qln * r / self.rmax
            jl1 = spherical_jn(l + 1, qln)
            factor = np.sqrt(2 * np.pi) / (jl1 * jl1 * self.rmax ** 3)
            dsbmat[:, j] = spherical_jn(l, arg) * factor
        return dsbmat

    #
    # r->q matrix transform
    #
    def dsBmatrix_inv(self, l, nq, qmin):

        dsbmatinv = np.zeros((self.nr, nq))
        r = np.arange(self.nr) * self.rmax / self.nr

        for j in np.arange(nq):
            qln = self.blqq.jnuzeros[l, j + 1]
            arg = qln * r / self.rmax
            jl1 = spherical_jn(l + 1, qln)
            factor = np.sqrt(2 * np.pi)  # / (jl1*jl1*self.rmax**3)

            if qln > (qmin * 2 * np.pi * self.rmax):
                dsbmatinv[:, j] = spherical_jn(l, arg) * np.sqrt(2 / np.pi) * r * r * self.rmax * 1e30 / self.nr
            else:
                dsbmatinv[:, j] = 0.0

        return dsbmatinv.transpose()

    #
    # Convert the blrr matrices to the PADF (l->theta)
    #
    def Blrr_to_padf(self, blrr, padfshape):
        print(f'{len(blrr.l)=}')
        print(f'{blrr.l[0].data[:,:].shape=}')
        print(padfshape)
        padfout = np.zeros(padfshape)
        lmax = blrr.nl
        for l in np.arange(2, lmax):

            if (l % 2) != 0:
                continue

            s2 = padfout.shape[2]
            z = np.cos(2.0 * np.pi * np.arange(s2) / float(s2))
            Pn = legendre(int(l))
            p = Pn(z)

            print(f"DEBUG  {l=}  {blrr.l[l].nq=}")
            for i in np.arange(padfout.shape[0]):
                for j in np.arange(padfout.shape[1]):

                    padfout[i, j, :] += blrr.l[l].data[i, j] * p[:]

        return padfout
