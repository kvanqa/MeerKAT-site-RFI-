# MeerKAT HP of RFI

The project is aimed to statistical analyze the RFI eniviroment for MeerKAT telescope using RFI flags produced by cal and 
ingest rfi  pipeline. The probability of RFI occurrence will be computed as a function of frequency, time, baseline, elevation and azimuthal angle.

## Data

MeerKAT archive visibility data is going to be used for this study. The rdb data files in the archive will be analyzed.
Katdal library is used to open and manipualte the rdb files produced by MeerKAT telescope, katdal is the data access library.

## Requirements

- Numpy
- Katdal
- Pandas
- Dask
- Xarrays
- Numba

## Run the module

python kathprfi.py -v path_to_vis_files -f path_to_flag_files -z path/to/save/zarr_array/zarr_array_name.zarr -b path2save/badfiles/filename.npy -g path2save/goodfiles/goodfiles.npy
