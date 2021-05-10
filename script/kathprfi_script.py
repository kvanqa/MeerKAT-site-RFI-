#!/usr/bin/env python3
import argparse
import ast
import logging
import os
import six
import time as tme

import numpy as np
import pandas as pd
import xarray as xr

import kathprfi_single_file as kathp


def initialize_logs():
    """
    Initialize the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser(description='This package produces two 5-D arrays, '
                                                 'which are the counter array and the master array.'
                                                 'The arrays provides statistics about measured'
                                                 'RFI from MeerKAT telescope.')
    parser.add_argument('-c', '--config', action='store', type=str,
                       help='A config file that does subselction of data')
    parser.add_argument('-b', '--bad', action='store',  type=str,
                        help='Path to save list of bad files')
    parser.add_argument('-g', '--good', action='store', type=str, default='\tmp',
                        help='Path to save bad files')
    parser.add_argument('-z', '--zarr', action='store', type=str, default='\tmp',
                        help='path to save output zarr file')
    return parser


def main():
    # Initializing the log settings
    initialize_logs()
    logging.info('MEERKAT HISTORICAL PROBABILITY OF RADIO FREQUENCY INTERFERENCE FRAMEWORK')
    parser = create_parser()
    args = parser.parse_args()
    path2config = os.path.abspath(args.config)
    # Read in dictionary with keys and values from config file
    config = kathp.config2dic(path2config)
    # Get values from the dictionary
    filename = config['filename']
    name_col = config['name_col']
    corrpro = config['corrprod']
    scans = config['scan']
    flags = config['flag_type']
    pol = config['pol_to_use']
    dump_rate = int(config['dump_period'])
    correlator_mode = config['correlator_mode']
    if correlator_mode == '4k':
        freq_chan = 4096
    elif correlator_mode == '32k':
        freq_chan = 32000
    # Read in csv file with files to process
    data = pd.read_csv(filename)
    f = data[name_col].values
    badfiles = []
    goodfiles = []
    for i in range(len(f)):
        # Initializing 5-D arrays
        master = np.zeros((24, 4096, 2016, 8, 24), dtype=np.uint16)
        counter = np.zeros((24, 4096, 2016, 8, 24), dtype=np.uint16)
        s = tme.time()
        logging.info('Adding file {} : {}'.format(i, f[i]))
        try:
            pathvis = f[i]
            vis = kathp.readfile(pathvis)
            logging.info('File number {} has been read'.format(i))
            
            if len(vis.freqs) == freq_chan and vis.dump_period > (dump_rate-1) and vis.dump_period <= dump_rate:
                logging.info('Removing bad antennas')
                clean_ants = kathp.remove_bad_ants(vis)
                logging.info('Bad antennas has been removed.')
                good_flags = kathp.selection(vis, pol_to_use=pol, corrprod=corrpro, scan=scans,
                                             clean_ants=clean_ants, flag_type=flags)
                logging.info('Good flags has been returned')
                if good_flags.shape[0] * good_flags.shape[1] * good_flags.shape[2] != 0:
                    # Updating the array
                    ntime = good_flags.shape[0]
                    time_step = 1
                    if ntime <= time_step:
                        time_step = ntime
                    nant = 64
                    Bl_idx = kathp.get_bl_idx(vis, nant)
                    elbins = np.linspace(10, 80, 8)
                    azbins = np.arange(0, 360, 15)
                    el, az = kathp.get_az_and_el(vis)
                    logging.info('Start to update the master and counter array')
                    for tm in six.moves.range(0, ntime, time_step):
                        time_slice = slice(tm, tm + time_step)
                        flag_chunk = good_flags[time_slice].astype(int)
                        # average flags from 32k to 4k mode.
                        if correlator_mode == '32k':
                            flag_chuck = NewFlagChunk(flag_chunk)
                        Time_idx = kathp.get_time_idx(vis)[time_slice]
                        El_idx = kathp.get_el_idx(el, elbins)[time_slice]
                        Az_idx = kathp.get_az_idx(az, azbins)[time_slice]
                        master, counter = kathp.update_arrays(Time_idx, Bl_idx, El_idx, Az_idx,
                                                              flag_chunk, master, counter)
                    logging.info('{} s has been taken to update file number {}'.format(i,
                                                                                       tme.time()
                                                                                       - s))
                    goodfiles.append(f[i])
                    logging.info('Creating Xarray Dataset')
                    ds = xr.Dataset({'master': (('time', 'frequency', 'baseline', 'elevation',
                                                 'azimuth'), master),
                   'counter': (('time', 'frequency', 'baseline', 'elevation', 'azimuth'), counter)},
                   {'time': np.arange(24), 'frequency': vis.freqs, 'baseline': np.arange(2016),
                       'elevation': np.linspace(10, 80, 8), 'azimuth': np.arange(0, 360, 15)})
                    logging.info('Saving dataset')
                    name, ext = os.path.splitext(args.zarr)
                    flname = name+str(f[i][46:56])+ext
                    ds.to_zarr(flname, group='arr')
                    logging.info('Dataset has been saved')
                else:
                    logging.info('{} selection has a problem'.format(f[i]))
                    badfiles.append(f[i])
                    pass
            else:
                logging.info('Channel/dump has a problem')
                badfiles.append(f[i])
                pass
            np.save(args.good,goodfiles)
            np.save(args.bad,badfiles)
            logging.info('File has been saved')
            

        except Exception as e:
            logging.info(e)
            continue


if __name__=="__main__":
    main()
                   