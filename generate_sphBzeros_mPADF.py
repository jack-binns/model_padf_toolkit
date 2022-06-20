import numpy as np
import sphBzeros as sb
import sys
import os


#
# set the maximum l and number of zeros
#
def gen_sphBzeros(path, lmax=1, nt=1000):
    print(f'<gen_sphBzeros> Computing sphBzeros...')
    lmax = lmax
    nt = nt
    print(f'<gen_sphBzeros> lmax : {lmax}')
    print(f'<gen_sphBzeros> nt : {nt}')
    #
    # compute the zeros
    #
    z = sb.Jn_zeros(lmax, nt)

    outname = os.path.join(path, "sphbzeros_lmax" + str(lmax) + "_nt" + str(nt) + ".npy")
    np.save(outname, z)
    print(z)
