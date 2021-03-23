#!/usr/bin/env python3
import argparse
import logging
import os
import time as tme

import numpy as np
import pandas as pd

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
    parser.add_argument('-v', '--vis', action='store',  type=str,
                        help='Path to the csv file that cointains links to rdb files')
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
    logging.info('MEERKAT HISTORICAL PROBABILITY OF RADIO FREQUENCY INTERFERENCE FRAMEWORK.')
    parser = create_parser()
    args = parser.parse_args()
    path = os.path.abspath(args.vis)
    data = pd.read_csv(path)
    f = data['FullLink'].values
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
            if len(vis.freqs) == 4096 and vis.dump_period > 7 and vis.dump_period <= 8:
                logging.info('Removing bad antennas')
                clean_ants = kathp.remove_bad_ants(vis)
                logging.info('Bad antennas has been removed.')
                good_flags = kathp.selection(vis, pol_to_use='HH', corrprod='cross', scan='track',
                                             clean_ants=clean_ants, flag_type=['cal_rfi', 'ingest_rfi'])
                logging.info('Good flags has been returned')
                if good_flags.shape[0] * good_flags.shape[1] * good_flags.shape[2] != 0:
                    # create azimuth and elevation bins
                    azbins = get_az_idx(az, np.arange(0, 370, 15))
                    elbins = get_el_idx(el, np.arange(10, 90, 10))
                    # Updating the array
                    ntime = good_flags.shape[0]
                    time_step = 10
                    if ntime <= time_step:
                        time_step = ntime
                    for tm in six.moves.range(0, ntime, time_step):
                        time_slice = slice(tm, tm + time_step)
                        flag_chunk = good_flags[time_slice].astype(int)
                        master, counter = kathp.update_arrays(vis, time_slice, azbins, elbins, nant,
                                                              flag_chunk, Master, Counter)
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
                    ds.to_zarr(args.zarr+str(f[i]),'w')
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
                   