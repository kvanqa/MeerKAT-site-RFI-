# Radio-Frequency-Interference-Probability-Frame

The project is aimed to compute probability of Radio Frequency Interference(RFI) occupancy using RFI flags produced by cal and 
ingest rfi  pipeline. The probability of RFI occurrence will be computed as a function of frequency, time, baseline, elevation and
azimuthal angle.

## Data

MeerKAT archive visibility data is going to be used for this study. The rdb data files in the archive will be analyzed.
Katdal library is used to open the H5 files produced by MeerKAT telescope, katdal is the data access library.

## Requirment

- Numpy
- Katdal
- Pandas
- Dask
