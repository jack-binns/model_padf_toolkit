import numpy as np
import paramsFILT as params
import padflibdev as pld
import array
import os

#
# set up parameter class
#
p = params.paramsFILT()

#
# Read input parameters from a file
#
p.read_config_file()

#
# Set up an instance of the padf class
#
print("DEBUG pyblfilter.py qmax, rmax", p.qmax, p.rmax)
padf = pld.padfcl(nl=p.nl, nr=p.nr, nq=p.nq, qmin=p.qmin, qmax=p.qmax, rmax=p.rmax)

#
#  Calculate the filter files
#
print(p.nl, "- nl")
print(" ")
print(" ")
for l in range(p.nlmin, p.nl, 1):
    nmax = padf.blqq.sphB_samp_nmax(l, p.rmax, p.qmax)
    print("l nmax", l, nmax)

    dsB = padf.dsBmatrix(l, nmax)
    dsBinv = padf.dsBmatrix_inv(l, nmax, p.qmin)

    filtermat = np.dot(dsB, dsBinv)

    outname = p.outpath + p.tag + "_l" + str(l) + "_filter.npy"
    np.save(outname, filtermat)

    outname = p.outpath + p.tag + "_l" + str(l) + "_dsB.npy"
    np.save(outname, dsB)
