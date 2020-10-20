import sys
import numpy as np
import laspy as lp

from time import time
from scipy.spatial import cKDTree as kdtree
from matplotlib import pyplot as pl

def load_las(fn):
    with lp.file.File(fn) as fp:
        pt = np.c_[fp.x, fp.y, fp.z]

    return pt

def time_kdtree(pt, k, leafsize, n_jobs):
    t0 = time()
    tr = kdtree(pt, leafsize = leafsize)
    t1 = time()
    tr.query(pt, k = k, n_jobs = n_jobs)
    t2 = time()
    return (t1 - t0, t2 - t1)

if __name__ == '__main__':
    fn = sys.argv[0][:-8] + '.las'
    pt = load_las(fn)
    
    nreps = 10
    st = np.zeros((nreps, 2))
    kr = [10, 50, 100]
    lr = [4, 8, 16]
    jr = [1, 2, 4, 8]
    dt = np.zeros((len(kr), len(lr), len(jr), 4))
    for ki, k in enumerate(kr):
        for li, l in enumerate(lr):
            for ij, j in enumerate(jr):
                for i in range(nreps):
                    s, t = time_kdtree(pt, k, l, j)
                    rs = (k, l, j, i, s, t)
                    print('k=%04d, leafsize=%2d, n_jobs=%02d, repeat=%i, t_create=%.4f, t_query=%.3f' % rs)
                    st[i, 0] = s
                    st[i, 1] = t
                dt[ki, li, ij, :] = (st[:,0].mean(), st[:,0].std(), st[:,1].mean(), st[:,1].std())

    fg, ax = pl.subplots(1, 2, figsize = (19.2, 10.8))
    ax[0].set_title('create')
    ax[1].set_title('query')
    for ki, k in enumerate(kr):
        for li, l in enumerate(lr):
            dtcm = dt[ki, li, :, 0]
            dtcs = dt[ki, li, :, 1]
            dtqm = dt[ki, li, :, 2]
            dtqs = dt[ki, li, :, 3]
            ax[0].plot(jr, dtcm, 'o-', mfc = 'none', label = 'k=%i, leafsize=%i' % (k, l))
            ax[0].fill_between(jr, dtcm - dtcs, dtcm + dtcs, alpha = 0.2)
            ax[1].plot(jr, dtqm, 'o-', mfc = 'none', label = 'k=%i, leafsize=%i' % (k, l))
            ax[1].fill_between(jr, dtqm - dtqs, dtqm + dtqs, alpha = 0.2)

    ax[0].set_xlabel('number of threads')
    ax[1].set_xlabel('number of threads')
    ax[0].set_ylabel('time [s]')
    ax[1].set_ylabel('time [s]')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    pl.savefig('%s.png' % sys.argv[0][:-3])
