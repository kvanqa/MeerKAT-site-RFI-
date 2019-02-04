# Statistical Analysis of Radio Frequency Interference Environment  for MeerKAT Telescope in the L-band Spectrum

The project is aimed to compute probability of Radio Frequency Interference(RFI) occupancy using RFI flags produced by cal and 
ingest rfi  pipeline. The probability of RFI occurrence will be computed as a function of frequency, time, baseline, elevation and
azimuthal angle.

## Data

MeerKAT archive visibility data is going to be used for this study. The rdb data files in the archive will be analyzed.
Katdal library is used to open and manipualte the rdb files produced by MeerKAT telescope, katdal is the data access library.

## Requirements

- Numpy
- Katdal
- Pandas
- Dask
- Xarrays
