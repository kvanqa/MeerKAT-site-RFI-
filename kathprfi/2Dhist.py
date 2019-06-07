import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import pandas as pd
from dask import array as da
import dask as npy
import katdal
import katdal
import six
from scipy.interpolate import griddata
import argparse
from datetime import datetime
from scipy.stats import beta as dbeta
import tqdm
from numba import jit,prange


data = xr.open_zarr('/data/isaac/FT_stack.zarr/')

m = data.master
c = data.counter

p = m.astype(float)/c.astype(float)

print('DAta has been read')

@jit(cache=True,nopython=True,parallel=True)
def binning(p_chunk,binns,bin_edges):
    '''
    r: a 2D array to be binned
    '''
    for c in prange(p_chunk.shape[0]):
        r = p_chunk[c,:]
        for i in range(len(bin_edges)-1):
		binns[c,i] += len(r[(bin_edges[i]<r) & (r<bin_edges[i+1])])
     
    return binns


def chunking(p,binns,bin_edges):
    new_bins = []
    for i in range(p.shape[0]):
        p_chunk = p[i,:,:].values
        print(i,' chunk has been created')
        new_bins.append(binning(p_chunk,binns,bin_edges))
    return np.array(new_bins)


bin_edges = np.arange(0,1.05,0.05)
binns = np.zeros((p.shape[1],20))

TwoD_bins = chunking(p,binns,bin_edges)
np.save('/data/isaac/2dhist.npy',TwoD_bins)
