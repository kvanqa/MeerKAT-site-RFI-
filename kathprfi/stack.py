import numpy as np
import matplotlib.pylab as plt
import xarray as xr
import pandas as pd
from dask import array as da
import dask as npy
import katdal
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import argparse
from datetime import datetime
from scipy.stats import beta as dbeta
from tqdm import tqdm
from numba import jit, prange


print('Reading data')

data = xr.open_zarr('/data/isaac/DR0_cal+ingestRFI.zarr/')


print('data has been read')

print('Creating stack m')
stackm = data.master.stack(z=['azimuth','baseline','elevation']).values

print('Creating stack c')
stackc = data.counter.stack(z=['azimuth','baseline','elevation']).values

print('Creating data array')
t = data.baseline.shape[0]*data.elevation.shape[0]*data.azimuth.shape[0]

np.save('/data/isaac/Freqm.npy',stackm)
np.save('/data/isaac/Freqc.npy',stackc) 
ds = xr.Dataset({'master':(('time','Frequency','total'),stackm),
                'counter':(('time','Frequency','total'),stackc)},
                {'time':np.arange(24),'Frequency':data.frequency.values,'total':np.arange(t)})

print('Saving the data set')
ds.to_zarr('/data/isaac/FT_stack.zarr','w')
print('Dataset has been saved')