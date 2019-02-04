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



def readfile(pathlitevis,pathfullvis,pathflag):
    '''
    Reading in the both the lite and full visibility file, and also the flag file.
    
    Arg : path2lite, path2full and path to the flag file
    '''
    vislite = katdal.open(pathlitevis)
    visfull = katdal.open(pathfullvis)
    flagfile =h5py.File(pathflag)
    
    return vislite,visfull,flagfile



def remove_bad_ants(fullvis):
    '''
    Input: Take a fullvis rdb file, called h5 here, which is open using:
        MyFile = "1543190535_sdp_l0.rdb"
        MyLiteVis = katdal.open(MyFile)
        h5 = MyLiteVis.select(scans='track')
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



def select_and_apply_with_good_ants(litevis,fullvis,flagfile, pol_to_use, corrprod,scan,clean_ants):
    from dask import array as da
    '''
    This function is going to select correlation products and apply flag table
    to the lite visibility file.
        
    Arg: lite and full visibility, flagfile, pol to use, corrproducts and scan and good antennas.
    
    Output : The flag table with ingest rfi flags and cal rfi flags
    '''
    
    litevis.select(corrprods = corrprod, pol = pol_to_use, scans = scan,ants = clean_ants)
    fullvis.select(corrprods = corrprod, pol = pol_to_use, scans = scan,ants = clean_ants)
    
    
    flags = da.from_array(flagfile['flags'], chunks=(1, 342, litevis.shape[2]))
    litevis.source.data.flags = flags

    flag = litevis.flags[:, :, :]
    return flag


def get_az_and_el(fullvis):
    '''
    Getting the full the elevation and azimuth of the file.
    
    Arg: fullvis file
    
    Return: List of avaraged elevation and azimuth of all antennas per time stamp
    '''
    
    # Getting the azmuth and elevation
    az = fullvis.az
    el = fullvis.el
    
    azmean = (np.array([np.mean(az[:][i]) for i in range(az.shape[0])])).astype(int)
    elmean = (np.array([np.mean(el[:][i]) for i in range(el.shape[0])])).astype(int)
    
    return elmean,azmean


def get_time_idx(litevis):
    import datetime
    '''
    This function is going to convert unix time to hour of a day
    
    Input : h5 obeject
    
    Output : list with time dumps converted to hour of a day
    '''
    unix  = litevis.timestamps

    local_time = []
    for i in range(len(unix)):
        local_time.append(datetime.datetime.fromtimestamp((unix[i])).strftime('%H:%M:%S'))

    # Converting time to hour of a day
    hour = []
    for i in range(len(local_time)):
        hour.append(int(round(int(local_time[i][:2]) + int(local_time[i][3:5])/60 + float(local_time[i][-2:])/3600)))
    return np.array(hour)[None,:]


def get_az_idx(az,bins):
    '''
    This function is going get the index of the azimuth 
    
    Input : List of Azimuthal angle and azimuthal bins
    
    Output : Azimuthal index
    '''
    az_idx = []
    for az in az:
        for j in range(len(bins)-1):
            if bins[j] <= az < bins[j+1]:
                az_idx.append(j)
    
    return np.array(az_idx)[None,:]


def get_el_idx(el,bins):
    '''
    This function is going get the index of the elevation
    
    Input : List of elevation angle and bins
    
    Output : Elevation index
    
    '''
    el_idx = []
    for el in el:
        for j in range(len(bins)-1):
            if bins[j] <= el < bins[j+1]:
                el_idx.append(j+1)
    
    return np.array(el_idx)[None,:] 



def get_corrprods(litevis):
    '''
    This function is getting the corr products
    
    Input : Visibility file
    
    Output : Correlation products
    '''
    bl = litevis.corr_products
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
    A1, A2 = np.empty(nant*(nant-1)/2, dtype=np.int32), np.empty(nant*(nant-1)/2, dtype=np.int32)
    k = 0
    for i in range(nant):
        for j in range(i+1,nant):
            A1[k] = i
            A2[k] = j
            k += 1

    # Baseline antenna cobinations
    corr_products = np.array(['m{:03d}m{:03d}'.format(A1[i], A2[i]) for i in range(len(A1))])
    
    # Number of baselines
    nbl = (nant**2 - nant)/2


    df = pd.DataFrame(data=np.arange(nbl), index=corr_products).T
    
    bl_idx = df[corr_prods].values
    return (bl_idx[0])[:,None]

if __name__=='__main__':
    def get_files(path2flags,path2lite,path2full):
        path = [path2flags,path2lite,path2full]
        import os, fnmatch
        listOfflags = os.listdir(path[0]) 
        listOflite= os.listdir(path[1])  
        listOffull = os.listdir(path[2])
        patternflags = "*.h5"  
        patternlite = "*.rdb" 
        patternfull = "*.rdb" 
        dataflags = []
        datalite = []
        datafull =[]
        for entry in listOfflags:  
             if fnmatch.fnmatch(entry,patternflags):
                dataflags.append(entry[0:10])
        for entry in listOflite:  
             if fnmatch.fnmatch(entry, patternlite):
                datalite.append(entry[0:10])
        for entry in listOffull:  
             if fnmatch.fnmatch(entry, patternfull):
                datafull.append(entry[0:10])
        vis = list(set(datalite).intersection(set(datafull)))
        data = list(set(dataflags).intersection(set(vis)))

        litevis = []
        fullvis = []
        flags = []
        for i in range(len(data)):
            litevis.append(data[i]+'_sdp_l0.rdb')
            fullvis.append(data[i]+'_sdp_l0.full.rdb')
            flags.append(data[i]+'_sdp_l0_flags.h5')
        
        return flags,litevis,fullvis
   

    flag,l,f = get_files('/scratch2/isaac/rfi_data/3calImaging/flags','/scratch2/isaac/rfi_data/3calImaging','/scratch2/isaac/rfi_data/3calImagingfull')
    #Initializing the master array and the weghting
    master = np.zeros((24,4096,2016,10,24),dtype=np.uint16)
    counter = np.zeros((24,4096,2016,10,24),dtype=np.uint16)
    badfiles = []
    goodfiles = []
    for i in range(len(f)):
        try:
            print 'Adding file number ', i
            pathlitevis='/scratch2/isaac/rfi_data/3calImaging/'+l[i]
            pathfullvis='/scratch2/isaac/rfi_data/3calImagingfull/'+f[i]
            pathflag = '/scratch2/isaac/rfi_data/3calImaging/flags/'+flag[i]
            litevis,fullvis,flagfile = readfile(pathlitevis,pathfullvis,pathflag)
            print 'File ',i,'has been read'
            clean_ants = remove_bad_ants(fullvis)
            print 'good ants'
            good_flags = select_and_apply_with_good_ants(litevis,fullvis,flagfile,pol_to_use = 'HH',corrprod='cross',scan='track',
                                                        clean_ants = clean_ants)
            print 'Good flags'
            el,az = get_az_and_el(fullvis)
            time_idx = get_time_idx(litevis)
            az_idx = get_az_idx(az,np.arange(0,370,15))
            el_idx = get_el_idx(el,np.arange(0,100,10))
            print 'el and az extracted'
            corr_prods = get_corrprods(litevis)
            bl_idx= get_bl_idx(corr_prods,nant=64)
            print 'start to update master array' 
            # Updating the array
            master[time_idx,:,bl_idx,el_idx,az_idx] += np.transpose(good_flags, axes=[2,0,1])
            counter[time_idx,:,bl_idx,el_idx,az_idx] += 1
            print 'Master array has been updated'

            np.save('/scratch2/isaac/rfi_data/master_offline_flag.npy',master)
            np.save('/scratch2/isaac/rfi_data/counter_offline_flag.npy',counter)
            goodfiles.append(f[i])
        except Exception as e: 
            print(e)
            print f[i],'file has a problem'
            badfiles.append(f[i])
            pass

    np.save('/scratch2/isaac/rfi_data/badfiles_offline_flag.npy',badfiles)
    np.save('/scratch2/isaac/rfi_data/goodfiles_offline_flag.npy',goodfiles)