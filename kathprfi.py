#!/usr/bin/env python
# coding: utf-8
import katdal
import h5py
import numpy as np
import matplotlib as plt
import xarray as xr
import pandas as pd
import pylab as plt
import time as tme
from dask import array as da
from dask import delayed
from numba import jit, prange
import argparse,os
import six



def readfile(pathfullvis, pathflag):
    '''
    Reading in the full visibility file, and also the flag file.
    
    Arg : path2full and path to the flag file
    '''
   
    visfull = katdal.open(pathfullvis)
    flagfile = h5py.File(pathflag)
    
    return visfull, flagfile



def remove_bad_ants(fullvis):
    '''
    This function is going to extrcat the list of all goos antennas.
    
    Input: Take a fullvis rdb file
    
    Output: List of good antennas
    '''
    # This pull all the antenna used for observation
    AntList = []
    for ant in fullvis.ants:
        AntList.append(ant.name)
    
    # This will give the antenna activity list
    AntsActivity = []
    for AntName in AntList: 
        AntsActivity.append((AntName, fullvis.sensor[AntName+'_activity']))

    for i in range(len(AntsActivity)):
        if 'stop' in AntsActivity[i][1]:
            AntList.remove(AntsActivity[i][0])
        else:
            pass
    
    return AntList



def select_and_apply_with_good_ants(fullvis,flagfile, pol_to_use, corrprod,scan,clean_ants):
    from dask import array as da
    '''
    This function is going to select correlation products and apply flag table
    to the full visibility file.
        
    Arg: full visibility, flagfile, pol to use, corrproducts and scan and good antennas.
    
    Output : The flag table with ingest rfi flags and cal rfi flags
    '''
    
    fullvis.select(reset='TFB')
    flags = da.from_array(flagfile['flags'], chunks=(1, 342, fullvis.shape[2]))
    
    fullvis.source.data.flags = flags
    
    fullvis.select(corrprods=corrprod, pol=pol_to_use, scans=scan,ants = clean_ants,flags=['cal_rfi','ingest_rfi'])

    flag =fullvis.flags  
  
    
    return flag


def get_az_and_el(fullvis):
    '''
    Getting the full the elevation and azimuth of the file.
    
    Arg: fullvis file
    
    Return: List of avaraged elevation and azimuth of all antennas per time stamp
    '''
    
    # Getting the azmuth and elevation

    azmean = np.mean(fullvis.az, axis=1)%360
    elmean = np.mean(fullvis.el, axis=1)
  
    
    return elmean, azmean


def get_time_idx(fullvis):
    import datetime
    '''
    This function is going to convert unix time to hour of a day
    
    Input : full visibility data file object
    
    Output : list with time dumps converted to hour of a day
    '''
    unix  = fullvis.timestamps

    local_time = []
    for i in range(len(unix)):
        local_time.append(datetime.datetime.fromtimestamp((unix[i])).strftime('%H:%M:%S'))

    # Converting time to hour of a day
    hour = []
    for i in range(len(local_time)):
        hour.append(int(round(int(local_time[i][:2]) + int(local_time[i][3:5])/60 + float(local_time[i][-2:])/3600)))
    return np.array(hour, dtype=np.int32)


def get_az_idx(azimuth,bins):
    '''
    This function is going get the index of the azimuth 
    
    Input : List of Azimuthal angle and azimuthal bins
    
    Output : Azimuthal index
    '''
    az_idx = []
    for az in azimuth:
        for j in range(len(bins)-1):
            if bins[j] <= az < bins[j+1]:
                az_idx.append(j)
    
    return np.array(az_idx)


def get_el_idx(elevation,bins):
    '''
    This function is going get the index of the elevation
    
    Input : List of elevation angle and bins
    
    Output : Elevation index
    
    '''
    el_idx = []

    for el in elevation:
        for j in range(len(bins)):
            if bins[j] <= el < bins[j]+10:
                el_idx.append(j)
    
    return np.array(el_idx, dtype=np.int32)



def get_corrprods(fullvis):
    '''
    This function is getting the corr products
    
    Input : Visibility file
    
    Output : Correlation products
    '''
    bl = fullvis.corr_products
    bl_idx = []
    for i in range(len(bl)):
        bl_idx.append((bl[i][0][0:-1]+bl[i][1][0:-1]))
            
    return np.array(bl_idx)


def get_bl_idx(corr_prods,nant):
    '''
    This function is getting the index of the correlation products
    
    Input  : Correlation products, number of antennas
    
    Output : Baseline index
    '''
    nant = nant

    A1, A2 = np.triu_indices(nant, 1)

    # Baseline antenna cobinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    



    df = pd.DataFrame(data=np.arange(len(A1)), index=corr_products).T
    
    bl_idx = df[corr_prods].values[0].astype(np.int32)
    
    return bl_idx

def get_files(path2flags, path2full):
    '''
    This file is going to get the list of datafiles[flag files and full visibility]
    
    Input: path to: flagfiles and fullvis
    
    Ouput: List of flagfiles and fullvis names.
    '''
    path = [path2flags, path2full]
    import os, fnmatch
    listOfflags = os.listdir(path[0])  
    
    listOffull = os.listdir(path[1]) 
    
    patternflags = "*.h5"   
    patternfull = "*.rdb" 
    dataflags = []
    datafull =[]
    for entry in listOffull:  
        datafull.append(entry[0:10])
    for entry in listOfflags:  
         if fnmatch.fnmatch(entry,patternflags):
            dataflags.append(entry[0:10])
            
    data = list(set(datafull).intersection(set(dataflags)))
    
    fullvis = []
    flags = []
    for i in range(len(data)):
        fullvis.append(data[i]+'_sdp_l0.full.rdb')
        flags.append(data[i]+'_sdp_l0_flags.h5')
    
    return flags,fullvis

@jit(nopython=True, parallel=True)
def update_arrays(Time_idx, Bl_idx, El_idx, Az_idx, Good_flags, Master, Counter):
    '''
    This function is gonna update the master and counter array
    
    Input: time_idx, bl_idx, el_idx, az_idx, flags_array, master and counter arrays
    
    Output: update master and counter array
    '''
    cstep = 128
    cblocks = (4096 + cstep - 1) // cstep
    for cblock in prange(cblocks):
        c_start = cblock * cstep
        c_end = min(4096, c_start + cstep)
        for k in range(c_start, c_end):
            for i in range(len(Bl_idx)):
                for j in range(len(Time_idx)):
                    Master[Time_idx[j],k,Bl_idx[i],El_idx[j],Az_idx[j]] += Good_flags[j,k,i]
                    Counter[Time_idx[j],k,Bl_idx[i],El_idx[j],Az_idx[j]] += 1
                
            
    return Master, Counter



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This package produces two 5-D arrays, which are the counter array and the master array. The arrays provides statistics about measured RFI from MeerKAT telescope.',)
   
    parser.add_argument('-v',
                        '--vis', action='store',  type=str,
                    help='Path to the full rdb visibility files')
    parser.add_argument('-f',
                        '--flags',
                        action='store',  type=str,
                        help='Path to TOM flag files')
    parser.add_argument('-b',
                        '--bad', action='store',  type=str,
                    help='Path to save list of bad files')
    parser.add_argument('-g',
                        '--good', action='store', type=str,default = '\tmp',
                    help='Path to save bad files')
    parser.add_argument('-z',
                        '--zarr', action='store', type=str,default = '\tmp',
                    help='path to save output zarr file')
    parser.add_argument('-n','--no_of_files', action = 'store', type=int,
                        help='Multiple of number of files to save a number between 1 and 10',default=1 )
    
    args = parser.parse_args()

    
    #Getting the file names
    flag,f = get_files(args.flags,args.vis)
    
     
    #Initializing the master array and the weghting
    master = np.zeros((24,4096,2016,8,24), dtype=np.uint16) 
    counter =np.zeros((24,4096,2016,8,24), dtype=np.uint16) 
 
    # Running the Hp code
    badfiles = []
    goodfiles = []
    
    for i in range(0,1):
        print('Adding file {} : {}'.format(i, f[i]))
        try:
            pathfullvis=str(args.vis)+'/'+f[i]
            pathflag = str(args.flags)+flag[i]
            fullvis,flagfile = readfile(pathfullvis, pathflag)
            print('File ',i,'has been read')
        except Exception as e:
            print e
            pass
     
        
        if len(fullvis.freqs) == 4096:
            clean_ants = remove_bad_ants(fullvis)
            print('good ants')
            good_flags = select_and_apply_with_good_ants(fullvis, flagfile, pol_to_use='HH', corrprod='cross', scan='track', 
                                                         clean_ants=clean_ants)
            print('Good flags')
          
            if good_flags.shape[0]* good_flags.shape[1]* good_flags.shape[2]!= 0:
                
                el,az = get_az_and_el(fullvis)
                time_idx = get_time_idx(fullvis)
                az_idx = get_az_idx(az,np.arange(0,370,15))
                el_idx = get_el_idx(el,np.arange(10,90,10))
                print('el and az extracted')
                corr_prods = get_corrprods(fullvis)
                bl_idx = get_bl_idx(corr_prods, nant=64)
                # Updating the array
                s = tme.time()
                ntime = good_flags.shape[0]
                time_step = 8
                for tm in six.moves.range(0, ntime, time_step):
                    time_slice=slice(tm, tm + time_step)
                    flag_chunk = good_flags[time_slice].astype(int)
                    tm_chunk = time_idx[time_slice]
                    el_chunk = el_idx[time_slice]
                    az_chunk = az_idx[time_slice]
                    master, counter = update_arrays(tm_chunk, bl_idx, el_chunk, az_chunk, flag_chunk, master, counter)

                print(tme.time() - s)
                goodfiles.append(f[i])
                
            else:
                print(f[i],'selection has a problem')
                badfiles.append(f[i])
                pass
             
                   
        else:
            print(f[i],'channel has a problem')
            badfiles.append(f[i])
            pass
        
        
        if i%args.no_of_files==0 and i!=0:
        
        ds = xr.Dataset({'master': (('time','frequency','baseline','elevation','azimuth') , master),
                         'counter': (('time','frequency','baseline','elevation','azimuth'), counter)},
                        {'time': np.arange(24),'frequency':fullvis.freqs,'baseline':np.arange(2016),
                         'elevation':np.linspace(10,80,8),'azimuth':np.arange(0,360,15)})

        ds.to_zarr(args.zarr,'w')
        np.save(args.good,goodfiles)
        np.save(args.bad,badfiles)
       
