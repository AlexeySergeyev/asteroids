from astropy.io import fits, ascii
from astropy.time import Time
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.imcce import Skybot
import os
from astroquery.sdss import SDSS
import numpy as np
from astropy.visualization import make_lupton_rgb, simple_norm
import matplotlib.pyplot as plt
import matplotlib.ticker as tic
from astropy.table import Table, vstack
import time
import re
import pandas as pd
from itertools import islice
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
from photutils import SkyCircularAperture
from scipy.stats import gaussian_kde
# import scipy, scipy.stats, scipy.signal
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import tarfile
# import seaborn as sns
from matplotlib import patches
import gc
from collections import deque
import pickle
import requests
import ps1_load as ps1


# import matplotlib as mpl
# from urllib3.packages.six import print_


def get_mjd():
    hdu = fits.open('/media/gamer/Data/data/sdss/g/109/frame-g-000109-1-0011.fits.bz2')
    header = hdu[0].header
    print(header)
    w = wcs.WCS(header)
    print(w.wcs.mjdobs)

    temp = header['DATE-OBS'].split('/')
    if int(temp[2]) > 50:
        date_obs = f'19{temp[2]}-{temp[1]}-{temp[0]}'
    else:
        date_obs = f'20{temp[2]}-{temp[1]}-{temp[0]}'

    temp = header['TAIHMS'].split(':')
    time = float(temp[0]) + float(temp[1]) / 60 + float(temp[2]) / 3600.0
    hour = int(time)
    minute = int((time % 1) * 60)
    second = ((time * 60) % 1) * 60

    exp = float(header['EXPTIME'])
    mjdtime = (exp / 2) / 3600.0 / 24.0

    time_obs = f'{date_obs}T{hour:02d}:{minute:02d}:{second:2.2f}'
    epoch = Time(time_obs, format='isot')
    print(epoch.mjd)
    epoch_corr = Time(epoch.mjd + mjdtime, format='mjd')
    print(epoch_corr.mjd)

    hdu.close()


def load_adr4():
    adr4 = ascii.read('/media/gamer/Data/data/sdss/init/ADR4.dat')
    adr = Table([adr4['col1'], adr4['col2'], adr4['col8'], adr4['col9'], adr4['col10']],
                names=('name', 'run', 'mjd', 'ra', 'dec'))
    ascii.write(adr, '/media/gamer/Data/data/sdss/init/ADR4a.dat')
    print(adr[:10])


def check_adr4():
    run = 125
    adr4 = ascii.read('/media/gamer/Data/data/sdss/init/ADR4a.dat')
    adr4_run = adr4[adr4['run'] == run]
    adr4_coord = SkyCoord(adr4_run['ra'] * u.deg, adr4_run['dec'] * u.deg)
    print(len(adr4_run))

    scalarc = SkyCoord(52.5557 * u.deg, 0.8714 * u.deg)
    d2d = scalarc.separation(adr4_coord)
    # print(d2d.deg)
    catalogmsk = d2d < 1 * u.arcmin
    print(adr4_run[catalogmsk])


def test_adr4_known():
    run = 125
    skybot = Skybot()
    adr4 = ascii.read('/media/gamer/Data/data/sdss/init/ADR4a.dat')
    adr4_run = adr4[adr4['run'] == run]
    adr4_run.sort('ra')
    ascii.write(adr4_run, f'{path}init/{run}.dat')
    adr4_coord = SkyCoord(adr4_run['ra'] * u.deg, adr4_run['dec'] * u.deg)

    list_known = []
    with open('/media/gamer/Data/data/sdss/figs/125/known/files', 'rt') as f:
        for line in f:
            a = line.strip().split('_')
            # print(a)
            list_known.append([float(a[1]), float(a[2]), float(a[3])])
    known = np.array(list_known)
    known_coord = SkyCoord(known[:, 1] * u.deg, known[:, 2] * u.deg)

    max_sep = 10.0 * u.arcsec
    idx, d2d, d3d = known_coord.match_to_catalog_sky(adr4_coord)
    sep_constraint = d2d > max_sep
    print(len(known_coord[sep_constraint]))

    names = []
    for i, item in enumerate(known[sep_constraint][:]):
        item_coord = SkyCoord(item[1] * u.deg, item[2] * u.deg)
        field = item_coord
        time_obs = Time(item[0], format='mjd')
        print(i, item_coord.to_string('hmsdms'))
        print(field, time_obs.iso, time_obs.mjd)
        objects = ['0:-1']
        while int(objects[0].split(':')[1]) == -1:
            objects = skybot.cone_search(item_coord, 10 * u.arcsec, time_obs.iso, location='645',
                                         get_raw_response=True, cache=True)
            objects = objects.split('\n')
            # print(objects)
            time.sleep(0.1)
        name = objects[3].split('|')[1]
        names.append(name)

    print(len(names), len(known))

    known_not_adr4 = Table(known[sep_constraint], names=('mjd', 'ra', 'dec'))
    known_not_adr4['name'] = names
    known_not_adr4.sort('ra')
    print(known_not_adr4)

    ascii.write(known_not_adr4, f'{path}init/known_not_adr4_{run}.dat')

    # print(known_coord[:10])


def mysearch():
    skybot = Skybot()
    # field = SkyCoord(359.411611 * u.deg, -0.792595 * u.deg)
    # epoch_corr = Time(51081.283841, format='mjd')
    # ns_arcsec = 9 * u.arcsec
    # objects = skybot.cone_search(field, ns_arcsec, epoch_corr, location='645', get_raw_response=True)

    from astroquery.imcce import Miriade
    from astropy.time import Time
    epoch = Time('2019-01-01', format='iso')
    objects = Miriade.get_ephemerides('3552', epoch=epoch)
    print(objects)

    # objects = objects.split('\n')
    # for item in objects:
    #     print(item)
    # number_objects = int(objects[0][-1])
    a = """632	51081.283841	359.411611	-0.792595	 2002 RQ237 	7
632	51081.283841	359.451128	-0.780609	 Inasan 	8"""


def time_diff():
    path = '/media/gamer/Data/data/sdss/'
    run = 109
    path_g = f'{path}g/{run}/'
    path_r = f'{path}r/{run}/'
    file_list_g = os.listdir(path_g)
    file_list_g = [item for item in file_list_g if item[-3:] == 'bz2']
    file_list_r = os.listdir(path_r)
    file_list_r = [item for item in file_list_r if item[-3:] == 'bz2']

    f = open('./data/diff_gr_time.dat', 'wt')

    for i in range(0, len(file_list_g)):
        if i % 10 == 0:
            print(i)
        hdu_g = fits.open(path_g + file_list_g[i])
        hdu_r = fits.open(path_r + file_list_r[i])
        g_time = hdu_g[0].header['TAIHMS'].split(':')
        r_time = hdu_r[0].header['TAIHMS'].split(':')
        g_hours = float(g_time[0]) + float(g_time[1]) / 60.0 + float(g_time[2]) / 3600.0
        r_hours = float(r_time[0]) + float(r_time[1]) / 60.0 + float(r_time[2]) / 3600.0
        dt = r_hours - g_hours
        if dt > 24:
            dt -= 24

        f.write(f'{dt * 3600.0:.3f}\n')
        hdu_g.close()
        hdu_r.close()

    f.close()


def sdss_req():
    query1 = f"""SELECT objID, ra, dec, rowv, colv, sqrt( power(rowv,2) + power(colv, 2) ) as velocity
    FROM PhotoObjAll 
    WHERE run = 125 and rerun=301 and camcol={int(fname[7])} and field={fname[9:13]} and 
      ((flags & 0x0000000080000000) != 0)  -- moving object
      and (power(rowv,2) + power(colv, 2)) > 0.0025 
      and rowv != -9999
    order by ra"""

    print(query1)
    res1 = SDSS.query_sql(query1, 14)
    print(res1)
    ascii.write(res1, f'./data/{fname}.dat', overwrite=True)


def show_image(fname):
    hdu_g = fits.open(f'/media/gamer/Data/data/sdss/g/125/frame-g-{fname}.fits.bz2')
    hdu_r = fits.open(f'/media/gamer/Data/data/sdss/r/125/frame-r-{fname}.fits.bz2')
    hdu_i = fits.open(f'/media/gamer/Data/data/sdss/i/125/frame-i-{fname}.fits.bz2')

    cx_g, cy_g = 1024, 744.5
    w_g = wcs.WCS(hdu_g[0].header)
    w_r = wcs.WCS(hdu_r[0].header)
    w_i = wcs.WCS(hdu_i[0].header)
    cg = w_g.wcs_pix2world(np.array([[cx_g, cy_g]]), 0)
    print(cg[0][0], cg[0][1])
    c_r = w_r.wcs_world2pix([[cg[0][0], cg[0][1]]], 0)
    c_i = w_i.wcs_world2pix([[cg[0][0], cg[0][1]]], 0)
    # print(c_r, c_i)
    dy_r, dx_r = int(round(cx_g - c_r[0][0])), int(round(cy_g - c_r[0][1]))
    dy_i, dx_i = cx_g - c_i[0][0], cy_g - c_i[0][1]
    print(dx_r, dy_r, dx_i, dy_i)

    nx, ny = 20, 20
    data_g = hdu_g[0].data[nx:-nx, ny:-ny]
    data_r = hdu_r[0].data[nx - int(round(dx_r)):-nx - int(round(dx_r)),
             ny - int(round(dy_r)):-ny - int(round(dy_r))]
    data_i = hdu_i[0].data[nx - int(round(dx_i)):-nx - int(round(dx_i)),
             ny - int(round(dy_i)):-ny - int(round(dy_i))]
    print(data_g.shape, data_r.shape, data_i.shape)

    rgb_default = make_lupton_rgb(data_i, data_r, data_g * 1.5, Q=5, stretch=0.5)
    plt.figure(figsize=(14, 10))
    plt.imshow(rgb_default, origin='lower')
    moved = ascii.read(f'./data/{fname}.dat')
    # print(moved['ra'], moved['dec'])
    pos_wcs = np.transpose((moved['ra'], moved['dec']))
    moved_pix = w_g.wcs_world2pix(pos_wcs, 0)
    print(moved_pix[:, 0])
    plt.scatter(moved_pix[:, 0] - nx, moved_pix[:, 1] - ny, edgecolors='white', facecolors='none',
                alpha=0.5, lw=0.2)
    for i, txt in enumerate(moved['objID']):
        # print(txt-1237646012716740000)
        plt.annotate(txt - 1237646012171870000, (moved_pix[i, 0] - nx + 10, moved_pix[i, 1] - ny - 10), color='white')
    plt.savefig(f'./data/frame-c-{fname}.png', dpi=300)
    plt.show()


def check_runs():
    adr4 = ascii.read(f'{path}init/ADR4a.dat')
    adr4_runs = adr4['run']
    myset_adr4_set = set(adr4_runs)
    myset_adr4 = list(myset_adr4_set)
    myset_adr4.sort()
    # print(len(myset_adr4))
    print(f'ADR4 RUNs number:{len(myset_adr4)}')

    # t_adr4 = Table(names=('run', 'number', 'min', 'max', 'mean', 'date'),
    #                dtype=('i4', 'i4', 'f4', 'f4', 'f4', 'S24'))
    # t_adr4 = Table(names=('run', 'number'), dtype=('i4', 'i4'))
    # for run in myset_adr4[:]:
    #     # print(run)
    #     cur_run = adr4[adr4['run'] == run]['mjd']
    #     mean = np.mean(cur_run)
    #     epoch = Time(mean, format='mjd')
    #     t_adr4.add_row((run, len(cur_run), cur_run.min(), cur_run.max(), mean, epoch.iso))
    #     # t_adr4.add_row((run, len(cur_run)))
    # print(t_adr4[:10])
    # ascii.write(t_adr4, './data/adr4_runs.dat', overwrite=True)

    hdu = fits.open(f'{path}/init/window_flist.fits')
    table = hdu[1].data
    run = table['RUN']
    # print(run[:10])

    myset_sdss_set = set(run)
    myset_sdss = list(myset_sdss_set)
    myset_sdss.sort()
    print(f'SDSS DR16 RUNs number:{len(myset_sdss)}')
    # print(f'SDSS DR16 RUNs list:\n{myset_sdss}')

    dset_set = set(myset_sdss) & set(myset_adr4)
    dset = list(dset_set)
    dset.sort()
    print(f'Joint SDSS DR16 and ADR4 RUNs number:{len(dset_set)}')
    # print(f'Joint SDSS DR16 and ADR4 RUNs list:\n{dset_set}')
    # print(len(dset))
    # print(dset)

    # print()
    sdss_dset_set = myset_sdss_set - myset_adr4_set
    sdss_dset = list(sdss_dset_set)
    sdss_dset.sort()
    print(f'SDSS DR16 RUNs not in ADR4:{len(sdss_dset_set)}')
    # print(f'Joint SDSS DR16 - ADR4 RUNs list:\n{sdss_dset_set}')
    # print(sdss_dset)

    # print()
    adr4_dset_set = myset_adr4_set - dset_set
    adr4_dset = list(adr4_dset_set)
    adr4_dset.sort()
    print(f'ADR4 not in SDSS DR16 RUNs:{len(adr4_dset_set)}')
    print(adr4_dset)

    # ascii.write(Table([sdss_dset], names=['run']), './data/sdss_dset.csv', format='csv')
    # ascii.write(Table([adr4_dset], names=['run']), './data/adr4_dset.csv', format='csv')

    # t_sdss = Table(names=('run', 'number', 'min', 'max', 'mean', 'date'),
    #                dtype=('i4', 'i4', 'f4', 'f4', 'f4', 'S24'))
    # # t_sdss = Table(names=('run', 'number'), dtype=('i4', 'i4'))
    # for run in myset_sdss[:]:
    #     print(run)
    #     cur_run = table[table['RUN'] == run]['MJD']
    #     mean = np.mean(cur_run)
    #     epoch = Time(mean, format='mjd')
    #     # print(cur_run[:10])
    #     t_sdss.add_row((run, len(cur_run), cur_run.min(), cur_run.max(), np.mean(cur_run), epoch.iso))
    #     # t_sdss.add_row((run, len(cur_run)))
    # print(t_sdss[:10])
    # ascii.write(t_sdss, './data/sdss_runs.dat', overwrite=True)

    # print(len(myset_sdss))


def concat_tables():
    tables_path = '/media/gamer/Data/data/sdss/candidates/125/'
    file_list = os.listdir(tables_path)
    mytable = Table()
    for i, item in enumerate(file_list[:]):
        print(i, item)
        try:
            cur_table = Table(ascii.read(tables_path + item))
            # print(cur_table)
            mytable = vstack([mytable, cur_table])
            # print(table)
        except:
            print('Empty table')

    ascii.write(mytable, tables_path + '125.dat', overwrite=True)
    print(mytable)


def skybot_check():
    adr = pd.read_csv(f'{path}init/adr5a_sdss.csv')
    print('Load complete.')
    skybot = Skybot()

    n = 100
    for j in np.arange(0, 2):
        beg = j * n
        end = beg + n - 1
        adr5 = adr.loc[beg:end].reset_index(drop=True)
        print(f'Adr5 from {beg} to {end} selected')

        # add colunns
        adr5['sdss_iso_time'] = '1900-01-01 00:00:00.000'
        adr5['sdss_coords'] = '00h00m00 00d00m00.0000s'
        adr5['sdss_vel_arcsec_hour'] = -9999

        adr5['name'] = 'unknown'
        adr5['ra_skybot'] = -9999.0
        adr5['dec_skybot'] = -9999.0
        adr5['V'] = -9999.0
        adr5['ang_dist'] = -9999.0
        adr5['dv_ra'] = -9999.0
        adr5['dv_dec'] = -9999.0
        adr5['dv_abs'] = -9999.0
        print(f'initialising was done')

        max = adr5.__len__()
        for i in range(0, max):
            item_coord = SkyCoord(adr5.loc[i, 'ra'] * u.deg, adr5.loc[i, 'dec'] * u.deg)

            # tai to utc
            t1 = Time(adr5.loc[i, 'mjd_r'], format='mjd', scale='tai')
            time_obs = Time(t1.tai, format='mjd', scale='utc')

            num_asteroids = -1
            count_request = 0
            while (num_asteroids == -1) and (count_request < 10):
                if count_request == 0:
                    objects = skybot.cone_search(item_coord, 30 * u.arcsec, time_obs.iso, location='645',
                                                 get_raw_response=True, cache=False)
                else:
                    objects = skybot.cone_search(item_coord, 30 * u.arcsec, time_obs.iso, location='645',
                                                 get_raw_response=True, cache=False)
                # print(objects)
                objects = objects.split('\n')
                num_asteroids = int(objects[0].split(':')[1])

                adr5.loc[i, 'mjd'] = time_obs.mjd
                adr5.loc[i, 'sdss_iso_time'] = time_obs.iso
                adr5.loc[i, 'sdss_coords'] = SkyCoord(ra=adr5.loc[i, 'ra'], dec=adr5.loc[i, 'dec'],
                                                      unit="deg").to_string('hmsdms')
                adr5.loc[i, 'sdss_vel_arcsec_hour'] = adr5.loc[i, 'velocity'] * 3600.0 / 24.0

                if num_asteroids > 0:
                    adr5.loc[i, 'name'] = objects[3].split('|')[1]
                    adr5.loc[i, 'ra_skybot'] = objects[3].split('|')[2]
                    adr5.loc[i, 'dec_skybot'] = objects[3].split('|')[3]
                    # coord = SkyCoord(ra_know, dec_know, unit=(u.hourangle, u.deg))

                    adr5.loc[i, 'V'] = objects[3].split('|')[5]
                    adr5.loc[i, 'ang_dist'] = objects[3].split('|')[7]
                    adr5.loc[i, 'dv_ra'] = objects[3].split('|')[8]  # arcsec/h
                    adr5.loc[i, 'dv_dec'] = objects[3].split('|')[9]  # arcsec/h
                    adr5.loc[i, 'dv_abs'] = np.sqrt(
                        float(adr5.loc[i, 'dv_ra']) ** 2 + float(adr5.loc[i, 'dv_dec']) ** 2)
                else:
                    count_request += 1
                time.sleep(0.1)
            print(i, adr5.loc[i, 'id'], f"{adr5.loc[i, 'ra']:.6f}", f"{adr5.loc[i, 'dec']:.6f}",
                  f"{Time(adr5.loc[i, 'mjd_r'], format='mjd').iso}", num_asteroids, count_request)

        adr5['psfMag_u'] = adr5['psfMag_u'].map('{:,.5f}'.format)
        adr5['psfMag_g'] = adr5['psfMag_g'].map('{:,.5f}'.format)
        adr5['psfMag_r'] = adr5['psfMag_r'].map('{:,.5f}'.format)
        adr5['psfMag_i'] = adr5['psfMag_i'].map('{:,.5f}'.format)
        adr5['psfMag_z'] = adr5['psfMag_z'].map('{:,.5f}'.format)
        adr5['psfMagErr_u'] = adr5['psfMagErr_u'].map('{:,.5f}'.format)
        adr5['psfMagErr_g'] = adr5['psfMagErr_g'].map('{:,.5f}'.format)
        adr5['psfMagErr_r'] = adr5['psfMagErr_r'].map('{:,.5f}'.format)
        adr5['psfMagErr_i'] = adr5['psfMagErr_i'].map('{:,.5f}'.format)
        adr5['psfMagErr_z'] = adr5['psfMagErr_z'].map('{:,.5f}'.format)
        adr5['rowv'] = adr5['rowv'].map('{:,.8f}'.format)
        adr5['colv'] = adr5['colv'].map('{:,.8f}'.format)

        # print(adr5['psfMag_u'])
        adr5.to_csv(f'{path}init/adr5_test_{int(beg // n):03d}_{int((end + 1) // n):03d}.csv', index=False)


def skybot_check_known():
    adr = pd.read_csv(f'{path}init/adr5f3_full.csv')
    for j in np.arange(0, 1):
        beg = j * 10000
        end = beg + 10000
        adr5 = adr.loc[beg + 0:end].reset_index(drop=True)
        # print(adr5.__len__())

        # for i in range(0, 5):
        #     cur_run = adr5.loc[i]
        #     print(cur_run['id'], cur_run['ra'], cur_run['dec'])
        # print(adr5.loc[i, 'id'], adr5.loc[i, 'ra'], adr5.loc[i, 'dec'])

        # names_list = []
        # names_list.extend(list(adr5.columns))
        # dtype_list = ['<i8', '<i8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8']
        # names_list.extend(['name', 'ra_known', 'dec_known', 'V', 'ang_dist', 'dv_ra', 'dv_dec'])

        skybot = Skybot()
        max = adr5.__len__()
        # mjdtime = 27 / 3600 / 24
        # for item in adr5[:10]:
        for i in range(0, max):
            # cur_row = pd.DataFrame(adr5.loc[i])
            # print(i, adr5.loc[i, 'id'], adr5.loc[i, 'ra'], adr5.loc[i, 'dec'], adr5.loc[i, 'mjd_r'])
            # print(adr5.loc[i, 'id'], adr5.loc[i, 'ra'], adr5.loc[i, 'dec'], adr5.loc[i, 'dec'])
            item_coord = SkyCoord(adr5.loc[i, 'ra'] * u.deg, adr5.loc[i, 'dec'] * u.deg)

            # tai to utc
            t1 = Time(adr5.loc[i, 'mjd_r'], format='mjd', scale='tai')
            time_obs = Time(t1.tai, format='mjd', scale='utc')

            # objects = ['0:-1']
            num_asteroids = -1
            count_request = 0
            while (num_asteroids == -1) and (count_request < 10):
                objects = skybot.cone_search(item_coord, 30 * u.arcsec, time_obs.iso, location='645',
                                             get_raw_response=True, cache=False)
                print(objects)
                objects = objects.split('\n')
                num_asteroids = int(objects[0].split(':')[1])
                if num_asteroids > 0:
                    adr5.loc[i, 'name'] = objects[3].split('|')[1]
                    adr5.loc[i, 'ra_skybot'] = objects[3].split('|')[2]
                    adr5.loc[i, 'dec_skybot'] = objects[3].split('|')[3]
                    # coord = SkyCoord(ra_know, dec_know, unit=(u.hourangle, u.deg))

                    adr5.loc[i, 'V'] = objects[3].split('|')[5]
                    adr5.loc[i, 'ang_dist'] = objects[3].split('|')[7]
                    adr5.loc[i, 'dv_ra'] = objects[3].split('|')[8]  # arcsec/h
                    adr5.loc[i, 'dv_dec'] = objects[3].split('|')[9]  # arcsec/h
                    adr5.loc[i, 'dv_abs'] = np.sqrt(
                        float(adr5.loc[i, 'dv_ra']) ** 2 + float(adr5.loc[i, 'dv_dec']) ** 2)
                    adr5.loc[i, 'iso'] = Time(adr5.loc[i, 'mjd_r'], format='mjd').iso
                    adr5.loc[i, 'sdss_coords'] = SkyCoord(ra=adr5.loc[i, 'ra'], dec=adr5.loc[i, 'dec'],
                                                          unit="deg").to_string('hmsdms')
                    adr5.loc[i, 'sdss_vel_arcsec_hour'] = adr5.loc[i, 'velocity'] * 3600.0 / 24.0
                elif num_asteroids == 0:
                    adr5.loc[i, 'name'] = 'unknown'
                else:
                    count_request += 1
                time.sleep(0.1)
            print(i, adr5.loc[i, 'id'], f"{adr5.loc[i, 'ra']:.6f}", f"{adr5.loc[i, 'dec']:.6f}",
                  f"{Time(adr5.loc[i, 'mjd_r'], format='mjd').iso}", num_asteroids, count_request)
        adr5.to_csv(f'{path}init/adr5d_known_{int(beg // 10000):02d}_{int(end // 10000):02d}.csv', index=False)
        # print(len(adr5['name'] == 'unknown'))


def check_run125():
    run = 125
    ns = 32
    adr4_125 = pd.read_csv(f'{path}init/adr4_run125.csv')
    l = adr4_125.__len__()
    print(l)
    for i in range(100, l):
        # camcol = adr4_125.iloc[i]['camcol']
        # field = adr4_125.iloc[i]['field']
        filename_g = f'{path}g/{run}/frame-g-{int(run):06d}-{int(adr4_125.iloc[i]["camcol"]):1d}' \
                     f'-{int(adr4_125.iloc[i]["field"]):04d}.fits.bz2'
        filename_r = f'{path}r/{run}/frame-r-{int(run):06d}-{int(adr4_125.iloc[i]["camcol"]):1d}' \
                     f'-{int(adr4_125.iloc[i]["field"]):04d}.fits.bz2'
        filename_i = f'{path}i/{run}/frame-i-{int(run):06d}-{int(adr4_125.iloc[i]["camcol"]):1d}' \
                     f'-{int(adr4_125.iloc[i]["field"]):04d}.fits.bz2'
        ra, dec = adr4_125.iloc[i]['ra'], adr4_125.iloc[i]['dec']
        print(filename_g)
        hdu_g = fits.open(filename_g)
        hdu_r = fits.open(filename_r)
        hdu_i = fits.open(filename_i)

        w_g = wcs.WCS(hdu_g[0].header)
        w_r = wcs.WCS(hdu_r[0].header)
        w_i = wcs.WCS(hdu_i[0].header)

        c_g_xy = w_g.wcs_world2pix([[ra, dec]], 0)
        c_r_xy = w_r.wcs_world2pix([[ra, dec]], 0)
        c_i_xy = w_i.wcs_world2pix([[ra, dec]], 0)

        data_g = hdu_g[0].data[int(c_g_xy[0][1] - ns):int(c_g_xy[0][1] + ns),
                 int(c_g_xy[0][0] - ns):int(c_g_xy[0][0] + ns)]
        data_r = hdu_r[0].data[int(c_r_xy[0][1] - ns):int(c_r_xy[0][1] + ns),
                 int(c_r_xy[0][0] - ns):int(c_r_xy[0][0] + ns)]
        data_i = hdu_i[0].data[int(c_i_xy[0][1] - ns):int(c_i_xy[0][1] + ns),
                 int(c_i_xy[0][0] - ns):int(c_i_xy[0][0] + ns)]

        if data_g.shape == (ns * 2, ns * 2) and data_r.shape == (ns * 2, ns * 2) and data_i.shape == (ns * 2, ns * 2):
            rgb_default = make_lupton_rgb(data_i, data_r, data_g * 1.5, Q=5, stretch=0.2)
            plt.figure(figsize=(4, 4))
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            plt.imshow(rgb_default, origin='lower')
            plt.title(f'mjd={adr4_125.iloc[i]["mjd_r"]:.4f}')
            filename_fig = f'{path}figs/125/adr4/{int(i):06d}_' \
                           f'{adr4_125.iloc[i]["mjd_r"]:.4f}_{ra:.4f}_{dec:.4f}.jpeg'
            plt.savefig(filename_fig, dpi=150, overvrite=True)

        hdu_g.close()
        hdu_r.close()
        hdu_i.close()


def adr5_analysis():
    save = False
    if save:
        adr5 = pd.read_csv(f'{path}init/adr5_known_00_01.csv')
        print(adr5.info())
        # print(adr5[['psfMag_r', 'V']])
        known_adr5 = adr5[adr5['name'] != 'unknown']

        print(known_adr5.info())
        good_known_adr5 = adr5[(adr5['psfMag_g'] > 10) & (adr5['V'] > 10)]

        good_known_adr5['dv_abs'] = np.sqrt(good_known_adr5['dv_ra'] ** 2 + good_known_adr5['dv_dec'] ** 2)
        good_known_adr5['iso'] = Time(good_known_adr5['mjd_r'], format='mjd').iso
        good_known_adr5['coords'] = SkyCoord(ra=good_known_adr5['ra'], dec=good_known_adr5['dec'],
                                             unit="deg").to_string(
            'hmsdms')
        good_known_adr5['sdss_vel_arcsec_hour'] = good_known_adr5['velocity'] * 3600.0 / 24.0
        good_known_adr5.to_csv('/media/gamer/Data/data/sdss/init/adr5_known.csv', index=False)
    else:
        good_known_adr5 = pd.read_csv(f'{path}init/adr5_full.csv')
        good_known_adr5 = good_known_adr5[good_known_adr5['name'] != 'unknown']
        good_known_adr5.info()

    x = good_known_adr5['psfMag_g'].to_numpy()
    y = good_known_adr5['V'].to_numpy()
    print(y.shape)

    xx = np.linspace(x.min(), 24, 10)
    fit = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=5.0)
    line_init = models.Linear1D()
    fitted_line, mask = or_fit(line_init, x, y)
    good_known_adr5['mask'] = mask

    # # Magnitude difference
    # xf = x[mask]
    # yf = y[mask]
    # print(xf.shape)
    # print(fitted_line)
    # # filtered_data = np.ma.masked_array(y, mask=mask)
    # plt.scatter(x=x, y=y, alpha=0.5, s=5.0, edgecolors='none')
    # plt.scatter(x=xf, y=yf, alpha=0.5, s=5.0, edgecolors='none')
    # plt.plot(xx, fitted_line(xx), color='black')
    # plt.title('Sigma clipped magnitude relation')
    # plt.annotate(f'y = {fitted_line.slope.value:.2f} * x + {fitted_line.intercept.value:.2f}', xy=(15, 23))
    # # plt.scatter(x=x, y=filtered_data, color='red', alpha=0.2, s=1.0)
    # plt.xlabel('psfMag_g (mag)')
    # plt.ylabel('V (mag)')
    # plt.grid(True)
    # # plt.savefig('./figs/magnitude.png', dpi=150)
    # # good_known_adr5.plot(kind='scatter', x='psfMag_g', y='V', color='blue', alpha=0.2, s=1.0)
    # plt.show()

    # # angular distance
    # ang = good_known_adr5['ang_dist'].to_numpy()
    # filtered_ang = np.ma.masked_array(ang, mask=mask)
    #
    # plt.hist(filtered_ang, bins=np.linspace(0, 5, 50), alpha=0.5, label='sdss')
    # plt.title('Angular distance between SDSS and MPC position')
    # plt.xlabel('angular distance (arcsec)')
    # # plt.savefig('./figs/distance.png', dpi=150)
    # plt.legend()
    # plt.show()
    #
    # # velocity
    # x = good_known_adr5['dv_abs'].to_numpy()
    # x_masked = np.ma.masked_array(x, mask=mask)
    # y = good_known_adr5['sdss_vel_arcsec_hour'].to_numpy()
    # y_masked = np.ma.masked_array(y, mask=mask)
    # plt.scatter(x=x, y=y, alpha=0.5, s=5.0, edgecolors='none')
    # plt.scatter(x=x_masked, y=y_masked, alpha=0.5, s=5.0, edgecolors='none')
    # plt.plot([0, 80], [0, 80], color='black')
    # plt.xlim(0, 80)
    # plt.ylim(0, 80)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    # # good_known_adr5.plot(kind='scatter', x='dv_abs', y='sdss_vel_arcsec_hour', color='red', alpha=0.2, s=1.0)
    # plt.xlabel('MPC velocity (arcsec/hour)')
    # plt.ylabel('SDSS velocity (arcsec/hour)')
    # plt.title('Velocity relation')
    # # plt.savefig('./figs/velocity.png', dpi=150)
    # plt.show()

    g = good_known_adr5['psfMag_g'].to_numpy()
    r = good_known_adr5['psfMag_r'].to_numpy()
    i = good_known_adr5['psfMag_i'].to_numpy()
    mask_color = ((g > 10) & (r > 10) & (i > 10))
    g1 = g[mask_color].copy()
    r1 = r[mask_color].copy()
    i1 = i[mask_color].copy()

    g2 = g[mask].copy()
    r2 = r[mask].copy()
    i2 = i[mask].copy()
    plt.scatter(x=g1 - i1, y=g1 - r1, alpha=0.2, s=5.0, facecolors='none')
    plt.scatter(x=g2 - i2, y=g2 - r2, alpha=0.2, s=5.0, edgecolors='none')
    plt.xlim(-5, 10)
    plt.ylim(-5, 10)
    plt.show()


def concat_adr5():
    fout = open(f'{path}init/adr5_full.csv', 'at')
    for i in range(23, 24):
        with open(f'{path}init/adr5_{i:03d}_{i + 1:03d}.csv', 'rt') as f:
            a = f.readlines()
            if i == 0:
                head = a[0]
                fout.write(head)
        for item in a[1:]:
            fout.write(item)
    fout.close()


def convert_to_MPC3():
    df = pd.read_csv(f'{path}init/adr5e4_known.csv')
    e = ' '
    mpc_list = ['NNNNNPPPPPPPANNYYYY MM DD.ddddd HH MM SS.dd sDD MM SS.d          MM.M B      OOO']
    print(mpc_list[0])
    for i in range(0, df.__len__()):
        # for i in range(0, 5):
        if i % 1000 == 0:
            print(i)
        output_string = ''
        adr5_name = df.iloc[i]['mpc']
        lname = len(adr5_name)
        if adr5_name != '0000000':
            if lname == 5:
                output_string += f'{adr5_name:5s}{e * 7}'
            else:
                output_string += f'{e * 5}{adr5_name:7s}'

        if output_string != '':
            output_string += e * 2 + 'C'

            # print(i)
            for band in ['u', 'g', 'r', 'i', 'z']:
                output_band = output_string
                # calculate time
                t1 = Time(df.iloc[i][f'TAI{band}'], format='mjd', scale='tai')
                time_obs = Time(t1.tai, format='mjd', scale='utc')

                output_band += f"{time_obs.ymdhms['year']}{e}{time_obs.ymdhms['month']:02d}{e}"
                a = time_obs.ymdhms['day'] + time_obs.ymdhms['hour'] / 24.0 + \
                    time_obs.ymdhms['minute'] / (24 * 60) + time_obs.ymdhms['second'] / (24 * 60 * 60)
                output_band += f"{a:02.5f}"

                coords = SkyCoord(ra=df.iloc[i]['ra'], dec=df.iloc[i]['dec'], unit="deg")
                dc_ra = df.iloc[i][f'offsetRa_{band}'] / np.cos(coords.dec.rad)
                dc_dec = df.iloc[i][f'offsetDec_{band}']
                # print(dc_ra, dc_dec)
                # print(re.split('h|m|s', coords.ra.to_string(u.hour)))
                new_coords = SkyCoord(ra=coords.ra.arcsec + dc_ra, dec=coords.dec.arcsec + dc_dec, unit="arcsec")
                # coords.dec.arcsec += df.iloc[i][f'offsetDec_{band}']
                # coords.ra.arcsec += df.iloc[i][f'offsetRa_{band}'] / np.cos(coords.dec.rad)
                # print(re.split('h|m|s', new_coords.ra.to_string(u.hour)))

                cur_ra = re.split('h|m|s', new_coords.ra.to_string(u.hour))
                cur_dec = re.split('d|m|s', new_coords.dec.to_string(u.degree, alwayssign=True))
                output_band += f'{e}{int(cur_ra[0]):02d}{e}{int(cur_ra[1]):02d}{e}{float(cur_ra[2]):05.2f}' \
                               f'{e}{int(cur_dec[0]):+03d}{e}{int(cur_dec[1]):02d}{e}{float(cur_dec[2]):04.1f}'
                output_band += e * 10

                mag = float(df.iloc[i][f"psfMag_{band}"])
                if mag > 0:
                    output_band += f'{mag:4.1f}'
                    output_band += e
                    output_band += band
                    output_band += e * 6
                    output_band += '645'
                    mpc_list.append(output_band)
                    # print(output_band)

    # for item in mpc_list[:100]:
    #     print(item)

    with open(f'{path}init/adr5e4_known_mpc.dat', 'wt') as f:
        for item in mpc_list[:]:
            f.write(f'{item}\n')


def convert_to_MPC2(df):
    mpc_names = pd.read_csv(f'{path}init/mpc_names2.csv')
    # num_strung = np.array2string(np.arange(1, 81) % 10, formatter={'int': lambda x: "%d" % x})[1:-1].replace(" ", "").replace(
    #     "\n", '')
    help_string = 'NNNNNPPPPPPPANNYYYY MM DD.ddddd HH MM SS.dd sDD MM SS.d          MM.M B      OOO'
    # print(num_strung)
    # print(help_string)
    e = ' '
    mpc_list = [help_string]
    mpc_wrong = []
    print(df.head())
    # df['shortname'] = np.full(df.__len__(), '0000000')
    for i in range(0, df.__len__()):
        if i % 100 == 0:
            print(i)
        output_string = ''
        adr5_name = df.iloc[i]['name'].lstrip().rstrip()
        pmpn = mpc_names.loc[mpc_names['name'] == adr5_name]  # Packed minor planet number

        if len(pmpn) > 0:
            id = pmpn.index[0] + 1
            df.iloc[i, df.columns.get_loc('shortname')] = pmpn["number"].iloc[0]
            if id <= 545135:
                output_string += f'{pmpn["number"].iloc[0]:5s}{e * 7}'
            else:
                output_string += f'{e * 5}{pmpn["number"].iloc[0]:7s}'
        else:
            mpc_wrong.append((i, adr5_name))
            print(f'{i}, Asteroid {adr5_name} not found, comet ?')
            # output_string += f'{e*5}{pmpn["number"].iloc[0]:7s}'

        if output_string != '':
            output_string += e * 2 + 'C'

            coords_r = SkyCoord(ra=df.iloc[i]['ra'], dec=df.iloc[i]['dec'], unit="deg")

            tu = Time(df.iloc[i]['TAIu'], format='mjd', scale='tai')
            time_obs_u = Time(tu.tai, format='mjd', scale='utc')

            output_string += f"{time_obs_u.ymdhms['year']}{e}{time_obs_u.ymdhms['month']:02d}{e}"
            a = time_obs_u.ymdhms['day'] + time_obs_u.ymdhms['hour'] / 24.0 + \
                time_obs_u.ymdhms['minute'] / (24 * 60) + time_obs_u.ymdhms['second'] / (24 * 60 * 60)
            output_string += f"{a:02.5f}"

            cur_ra = re.split('h|m|s', coords_r.ra.to_string(u.hour))
            cur_dec = re.split('d|m|s', coords_r.dec.to_string(u.degree, alwayssign=True))
            output_string += f'{e}{int(cur_ra[0]):02d}{e}{int(cur_ra[1]):02d}{e}{float(cur_ra[2]):05.2f}' \
                             f'{e}{int(cur_dec[0]):+03d}{e}{int(cur_dec[1]):02d}{e}{float(cur_dec[2]):04.1f}'
            output_string += e * 10
            # print(i)
            for band in ['u', 'g', 'r', 'i', 'z']:
                mag = float(df.iloc[i][f"psfMag_{band}"])
                if mag > 0:
                    output = output_string + f'{mag:4.1f}'
                    output += e
                    output += band
                    output += e * 6
                    output += '645'
                    mpc_list.append(output)

    for item in mpc_list[:]:
        print(item)

    # df.to_csv(f'{path}init/adr5d_known.csv', index=False)
    # with open(f'{path}init/adr5d_wrong.csv', 'wt') as f:
    #     f.write('id,name\n')
    #     for item in mpc_wrong:
    #         f.write(f'{item[0]},{item[1]}\n')


def convert_to_MPC(df):
    df_len = df.__len__()
    df_len = 1
    # df[df['name'].split('').str.contains("-")]
    # print(df[df['name'].str.contains("-")]['name'])

    # name to Pname
    for i in range(0, 5000):
        name = df.iloc[i]['name'].split(' ')
        # print(name)
        if len(name) == 3:
            pass
        elif len(name) == 4:
            try:
                int(name[1])
            except:
                print(name)
            else:
                if '-' in name[2]:
                    pname = name[2][:1] + name[2][:-1] + name[1]
                else:
                    # name[1], name[2] = '1995', 'TA418'
                    pname = f'{chr(int(name[1]) // 100 - 19 + ord("J"))}' \
                            f'{name[1][2:]}{name[2][0]}'
                    if len(name[2]) < 3:
                        pname += '00'
                    else:
                        a = int(name[2][2:])
                        if a < 100:
                            pname += f'{a:02d}'
                        elif a < 360:
                            dec = ord('A') + (a // 10 - 10)
                            pname += chr(dec) + name[2][-1:]
                        else:
                            d = a // 10 - 36
                            dec = ord('a') + d
                            pname += chr(dec) + name[2][-1:]
                    pname += name[2][1]

        # print(pname)


def adr5_full():
    adr5 = pd.read_csv(f'{path}init/adr5ca_full.csv')
    known_adr5 = adr5[adr5['name'] != 'unknown'].reset_index(drop=True)

    # unknown_adr5 = adr5[adr5['name'] == 'unknown']
    # print(known_adr5.head())
    # print(known_adr5.__len__(), unknown_adr5.__len__())

    # tu = Time(known_adr5.iloc[0]['TAIu'], format='mjd', scale='tai')
    # time_obs_u = Time(tu.tai, format='mjd', scale='utc')
    # s = f"{time_obs_u.ymdhms['year']} {time_obs_u.ymdhms['month']} {time_obs_u.ymdhms['day']} "
    # a = time_obs_u.ymdhms['day'] + time_obs_u.ymdhms['hour']/24.0 + \
    #     time_obs_u.ymdhms['minute']/(24 * 60) + time_obs_u.ymdhms['second']/(24 * 60 * 60)
    # s += f"{time_obs_u.ymdhms['hour']} {time_obs_u.ymdhms['minute']} {a:.5f}"
    # print(s)
    # coords_r = SkyCoord(ra=known_adr5.iloc[0]['ra'], dec=known_adr5.iloc[0]['dec'], unit="deg")
    # cur_ra = re.split('h|m|s', coords_r.ra.to_string(u.hour))
    # cur_dec = re.split('d|m|s', coords_r.dec.to_string(u.degree, alwayssign=True))
    # print(cur_ra, cur_dec)
    # output_string = f'{int(cur_ra[0]):02d} {int(cur_ra[1]):02d} {float(cur_ra[2]):02.2f} ' \
    #                 f'{int(cur_dec[0]):+03d} {int(cur_dec[1]):02d} {float(cur_dec[2]):02.1f}'
    # print(output_string)
    # convert_to_MPC2(known_adr5)

    # known_adr5 = pd.read_csv(f'{path}init/adr5d_known.csv')
    # convert_to_MPC3(known_adr5)


def load_mpcorb():
    # mpc_dtypes = {‘a’: np.float64, ‘b’: np.int32}
    # mpc_dtypes = {"Desn": ('U7', 0), "H": (np.float, 9), "G": (np.float, 15), "Epoch":('U5', 21),
    #               "M": (np.float32, 27), 'temp': ('U144', 38)}
    # "   Epoch     M        Peri.      Node       Incl.       e            n           a        Reference #Obs #Opp    Arc    rms  Perts   Computer"
    mpc = pd.read_csv(f'{path}init/MPCORB.DAT', header=31)
    mpc.columns = ['temp']
    mpc['number'] = mpc['temp'].map(lambda x: x[0:7].rstrip())
    mpc['name'] = mpc['temp'].map(lambda x: x[175:194].rstrip())
    del mpc['temp']
    print(mpc.tail(10))
    mpc.to_csv(f'{path}init/mpc_names2.csv', index=False)


def add_short_name():
    df = pd.read_csv(f'{path}init/adr5e_full.csv')
    df_add = pd.read_csv(f'{path}init/mpc_names2.csv')
    df['name'] = df['name'].str.lstrip().str.rstrip()
    s1 = 'https://api.ssodnet.imcce.fr/quaero/1/sso?q='

    l = df.__len__()
    short_names = np.full(l, '0000000')
    for i in range(0, l):
        a = df.iloc[i]['name']
        add_series = df_add.loc[df_add['name'] == a]
        if add_series.__len__() != 1:
            print(i, a)
            # s2 = a.replace(' ', '%20')
            # print(s1 + s2)
            # r = urllib.request.urlopen(s1 + s2)
            # print(r.read())
        else:
            short_names[i] = add_series.iloc[0]['number']
        if i % 100 == 0:
            print(i)

    # print(short_names)
    df['mpc'] = short_names
    df = format_adr(df)
    df.to_csv(f'{path}init/adr5e4_full.csv', index=False)


def full_sdss_aseroid_check():
    skybot = Skybot()
    idx = 12
    im_scale = 0.396
    hdu = fits.open(f'{path}/init/window_flist.fits')
    table = hdu[1].data
    tai = table['TAI'] / 3600 / 24
    ra = table['RA']
    dec = table['DEC']
    print(table['RA'][:10])

    shift = 0.0 / 3600 / 24
    # tai to utc
    t1 = Time(tai[idx] - shift, format='mjd', scale='tai')
    time_obs = Time(t1.tai, format='mjd', scale='utc')
    coords = SkyCoord(ra=ra[idx], dec=dec[idx], unit="deg")
    # # print(time_obs[0].mjd, (time_obs[0].mjd - tai[0]) * 3600 * 24)
    # print(coords, time_obs)
    #

    ddec = 2048 * im_scale / 2.0 / 3600
    dra = 1489 * im_scale / 2.0 / 3600
    rad = 1.0 * np.sqrt(dra ** 2 + ddec ** 2) * u.deg
    print(coords, rad, time_obs[2].iso)

    try:
        objects = skybot.cone_search(coords, rad, time_obs[2].iso, location='645',
                                     get_raw_response=False, cache=True)
        if len(objects) > 0:
            print(len(objects))
            inside = []
            outside = []
            for item in objects:
                ddra = 1 * dra / np.cos(coords.dec.deg / 180.0 * np.pi)
                print(ddra)
                if (coords.ra.deg - ddra < item['RA'].value < coords.ra.deg + ddra) and \
                        (coords.dec.deg - ddec < item['DEC'].value < coords.dec.deg + ddec):

                    # print(f'{item["RA"]} {item["DEC"]} inside {coords}')
                    inside.append((item['RA'].value, item['DEC'].value))
                else:
                    # print(f'{item["RA"]} {item["DEC"]} outside {coords}')
                    outside.append((item['RA'].value, item['DEC'].value))

        fname = f'frame-r-{table["RUN"][idx]:06d}-{table["CAMCOL"][idx]:d}-{table["FIELD"][idx]:04d}.fits.bz2'
        hdu = fits.open(f'{path}r/{table["RUN"][idx]}/{fname}')
        w = wcs.WCS(hdu[0].header)
        data = hdu[0].data
        ins_pix = w.wcs_world2pix(np.array(inside), 1)
        out_pix = w.wcs_world2pix(np.array(outside), 1)

        ins_pix1 = w.wcs_world2pix(np.array(inside), 1)
        out_pix1 = w.wcs_world2pix(np.array(outside), 1)
        # print(ins_pix1, ins_pix, ins_pix1 - ins_pix)
        # print(out_pix1, out_pix, out_pix1 - out_pix)

        # print(ins_pix)
        # outs_pix = w.wcs.WCS.world_to_pixel(outside, 0)

        fig = plt.figure()
        fig.add_subplot(111, projection=w)
        # plt.imshow(hdu.data, origin='lower', cmap=plt.cm.viridis)
        plt.xlabel('RA')
        plt.ylabel('Dec')
        plt.grid(color='white')
        # print(inside[:, 1])
        # a_ins = SkyCircularAperture(SkyCoord(ra=inside[:, 0], dec=inside[:, 1], unit='deg'), 3.0 * u.arcsec)
        # a_outs = SkyCircularAperture(SkyCoord(ra=outside[:, 0], dec=outside[:, 1], unit='deg'), 3. * u.arcsec)

        norm = simple_norm(data, stretch='log')
        plt.imshow(data, origin='lower', norm=norm)
        # a_ins.to_pixel(wcs=w)
        # a_outs.to_pixel(wcs=w).plot(color='orange')
        if len(inside) > 0:
            print(ins_pix.__len__(), ins_pix)
            plt.scatter(ins_pix[:, 0], ins_pix[:, 1], edgecolors='orange', facecolors='none')
        if len(outside) > 0:
            plt.scatter(out_pix[:, 0], out_pix[:, 1], edgecolors='green', facecolors='none')
        plt.show()
    except:
        print('No Solar objects was found')


def full_sdss_aseroid_run():
    skybot = Skybot()

    im_scale = 0.396
    ddec = 2048 * im_scale / 2.0 / 3600
    dra = 1489 * im_scale / 2.0 / 3600
    rad = 1 * np.sqrt(dra ** 2 + ddec ** 2) * u.deg

    hdu = fits.open(f'{path}/init/window_flist.fits')
    table = hdu[1].data
    tai = table['TAI'] / 3600 / 24
    ra = table['RA']
    dec = table['DEC']

    # tai to utc
    for idx in range(2000, 10000):
        t1 = Time(tai[idx], format='mjd', scale='tai')
        time_obs = Time(t1.tai, format='mjd', scale='utc')
        coords = SkyCoord(ra=ra[idx], dec=dec[idx], unit="deg")

        ddra = 1.0 * dra / np.cos(coords.dec.deg / 180.0 * np.pi)
        table_name = f'frame-r-{table["RUN"][idx]:06d}-{table["CAMCOL"][idx]:d}-{table["FIELD"][idx]:04d}.csv'

        iso_coords = coords.ra.to_string(unit=u.hourangle, sep=' ', precision=2, pad=True) \
                     + coords.dec.to_string(sep=' ', precision=2, alwayssign=True, pad=True)

        trying = False
        objects = []
        # while trying == False:
        try:
            objects = skybot.cone_search(coords, rad, time_obs[2].iso, location='645',
                                         get_raw_response=False, cache=True)
        except:
            print(f'{idx} It was error load error')
            # print(objects)
            # print(coords, rad, time_obs[2].iso, iso_coords, rad * 3600)
            t = Table()
            ascii.write(t, output=f'{path}asteroids/{table_name}.dat', format='tab')
            time.sleep(0.1)
        else:
            trying = True
            check = np.zeros(len(objects), dtype=bool)

            for i, item in enumerate(objects):
                if (coords.ra.deg - ddra < item['RA'].value < coords.ra.deg + ddra) and \
                        (coords.dec.deg - ddec < item['DEC'].value < coords.dec.deg + ddec):
                    check[i] = 1

            inside = objects[check]
            print(len(objects), len(inside), item['RA'], item['DEC'])
            # print(inside)

            ascii.write(inside, output=f'{path}asteroids/{table_name}', format='csv')
            # pos = np.array([inside['RA'], inside['DEC']])
            # fname = f'frame-r-{table["RUN"][idx]:06d}-{table["CAMCOL"][idx]:d}-{table["FIELD"][idx]:04d}.fits.bz2'
            # hdu = fits.open(f'{path}r/{table["RUN"][idx]}/{fname}')
            # w = wcs.WCS(hdu[0].header)
            # data = hdu[0].data
            # ins_pix = w.wcs_world2pix(pos.T, 0)
            #
            # fig = plt.figure()
            # fig.add_subplot(111, projection=w)
            # plt.xlabel('RA')
            # plt.ylabel('Dec')
            # plt.grid(color='white')
            #
            # norm = simple_norm(data, stretch='log')
            # plt.imshow(data, origin='lower', norm=norm)
            # if len(inside) > 0:
            #     # print(ins_pix.__len__(), ins_pix)
            #     plt.scatter(ins_pix[:, 0], ins_pix[:, 1], edgecolors='orange', facecolors='none')
            #
            # plt.show()


def concat_sdss():
    beg = 10000
    curent_path = f'{path}asteroids/{beg}/'
    file_list = os.listdir(curent_path)
    with open(f'{path}asteroids/{beg}.csv', 'at') as fout:
        for file in file_list[:]:
            if file.endswith('.csv'):
                a = []
                with open(f'{curent_path}{file}', 'rt') as f:
                    for line in f:
                        a.append(line.strip())
                for item in a[1:]:
                    fout.write(f'{item},{file[:-4]}\n')


def sort_ard5c_as_full():
    adr5 = pd.read_csv(f'{path}init/adr5_full.csv')
    adr5_new = pd.read_csv(f'{path}init/adr5c.csv')

    # print(adr5['objID'])
    adr5_new = adr5_new.set_index('objID')
    adr5_new = adr5_new.reindex(index=adr5['objID'])
    adr5_new = adr5_new.reset_index()
    # print(adr5_new)
    adr5.insert(loc=3, column='rerun', value=adr5_new['rerun'])
    adr5.insert(loc=4, column='camcol', value=adr5_new['camcol'])
    adr5.insert(loc=5, column='field', value=adr5_new['field'])
    adr5.insert(loc=6, column='TAIu', value=adr5_new['TAIu'])
    adr5.insert(loc=7, column='TAIg', value=adr5_new['TAIg'])
    adr5.insert(loc=8, column='TAIr', value=adr5_new['TAIr'])
    adr5.insert(loc=9, column='TAIi', value=adr5_new['TAIi'])
    adr5.insert(loc=10, column='TAIz', value=adr5_new['TAIz'])
    del adr5['mjd_g']
    del adr5['mjd_r']

    print(adr5)
    adr5.to_csv(f'{path}init/adr5c_full.csv', index=False)


def adr5d_color():
    def density_scatter(x, y, ax=None, sort=True, bins=20):
        """
        Scatter plot colored by 2d histogram
        """
        # from matplotlib.colors import Normalize
        from scipy.interpolate import interpn
        # from matplotlib import cm

        if ax is None:
            fig, ax = plt.subplots()
        data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
        z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                    method="splinef2d", bounds_error=False)

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        s = ax.scatter(x, y, s=2, c=z, cmap='jet')

        # norm = Normalize(vmin=np.min(z), vmax=np.max(z))
        # plt.colorbar(s, ax=ax, cmap='jet')
        # cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax, cmap='jet')
        # ax.set_ylabel('Density')
        # plt.show()

        return ax

    from scipy.stats import gaussian_kde
    sun = {'u-g': 1.40, 'g-r': 0.45, 'r-i': 0.12, 'i-z': 0.04}
    adr5 = pd.read_csv(f'{path}init/adr5d_known.csv')
    adr5_good = adr5.loc[(adr5['psfMag_u'] > 0) & (adr5['psfMag_g'] > 0) & (adr5['psfMag_r'] > 0) &
                         (adr5['psfMag_i'] > 0) & (adr5['psfMag_z'] > 0)]
    # print(adr5_good['psfMag_g'] - adr5_good['psfMag_r'])

    # plt.scatter(adr5_good['psfMag_r'],
    #             adr5_good['psfMag_g'] - adr5_good['psfMag_i'], alpha=0.1, s=2)
    # # plt.scatter(sun['u-g'], sun['g-r'])
    # plt.xlabel('r')
    # plt.ylabel('g-r')
    # plt.show()
    #
    seq = 1
    np_r = adr5_good['psfMag_r'].to_numpy()[::seq]
    np_gr = (adr5_good['psfMag_g'] - adr5_good['psfMag_r']).to_numpy()[::seq]
    np_ug = (adr5_good['psfMag_u'] - adr5_good['psfMag_g']).to_numpy()[::seq]
    np_ri = (adr5_good['psfMag_r'] - adr5_good['psfMag_i']).to_numpy()[::seq]
    np_iz = (adr5_good['psfMag_i'] - adr5_good['psfMag_z']).to_numpy()[::seq]

    # density_scatter(np_r, np_gr, bins=[50, 50])
    # plt.xlabel('r, (mag)')
    # plt.ylabel('g-r, (mag)')
    # # plt.xlim(15, 7.5)
    # # plt.ylim(-4, 11)
    # # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('./figs/color_r_gr.png', dpi=300)
    # plt.show()

    # density_scatter(np_ug, np_gr, bins=[50, 50])
    # plt.scatter(sun['u-g'], sun['g-r'], color='black', marker='+')
    # plt.xlabel('u-g, (mag)')
    # plt.ylabel('g-r, (mag)')
    # plt.xlim(-7.5, 7.5)
    # plt.ylim(-4, 11)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig('./figs/color_ug_gr.png', dpi=300)
    # plt.show()

    # density_scatter(np_ug, np_iz, bins=[50, 50])
    # plt.scatter(sun['u-g'], sun['i-z'], color='black', marker='+')
    # plt.xlim(-7.5, 7.5)
    # plt.ylim(-7.5, 7.5)
    # plt.gca().set_aspect('equal', adjustable='box')
    # # plt.axis('scaled')
    # # plt.grid(True)
    # plt.xlabel('u-g, (mag)')
    # plt.ylabel('i-z, (mag)')
    # plt.savefig('./figs/color_ug_iz.png', dpi=300)
    # plt.show()

    density_scatter(np_gr, np_ri, bins=[50, 50])
    plt.scatter(sun['g-r'], sun['r-i'], color='black', marker='+')
    plt.xlim(-4, 11)
    plt.ylim(-7.5, 7.5)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    plt.xlabel('g-r, (mag)')
    plt.ylabel('r-i, (mag)')
    plt.savefig('./figs/color_gr_ri.png', dpi=300)
    plt.show()

    #
    # # Calculate the point density
    # xy = np.vstack([np_r, np_ug])
    # z = gaussian_kde(xy)(xy)
    # print('A gaussian kde calculated')

    # fig, ax = plt.subplots()
    # ax.scatter(np_r, np_ug, c=z, s=5, edgecolor='', cmap='jet')
    # # plt.scatter(sun['u-g'], sun['g-r'], edgecolors='', facecolors='black')
    # plt.xlabel('r, (mag)')
    # plt.ylabel('g-r, (mag)')
    # plt.savefig('./data/color_r_gr.png', dpi=300)
    # plt.show()

    # # histogram the data
    # bins = [100, 100]
    # hh, locx, locy = np.histogram2d(np_r, np_ug, bins=bins)
    #
    # # Sort the points by density, so that the densest points are plotted last
    # z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(np_r, np_ug)])
    # idx = z.argsort()
    # x2, y2, z2 = np_r[idx], np_ug[idx], z[idx]
    #
    # # plt.figure(1, figsize=(8, 10)).clf()
    # s = plt.scatter(x2, y2, c=z, cmap='jet', marker='.')
    # plt.colorbar(s)
    # plt.show()
    #
    # plt.scatter(np_ug, np_gr, alpha=0.1, s=2)
    # plt.scatter(sun['u-g'], sun['g-r'])
    # plt.xlabel('u-g')
    # plt.ylabel('g-r')
    # plt.show()
    #
    # np_ug = (adr5_good['psfMag_u'] - adr5_good['psfMag_g']).to_numpy()[::5]
    # np_gr = (adr5_good['psfMag_g'] - adr5_good['psfMag_r']).to_numpy()[::5]
    # # h = plt.hist2d(np_ug, np_gr, bins=(50, 50), cmap='jet')
    # # histogram the data
    # bins = [10, 10]
    # hh, locx, locy = np.histogram2d(np_ug, np_gr, bins=bins)
    #
    # # Sort the points by density, so that the densest points are plotted last
    # z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])]
    #               for a, b in zip(np_ug, np_gr)])
    # idx = z.argsort()
    # x2, y2, z2 = np_ug[idx], np_gr[idx], z[idx]
    #
    # # plt.figure(1, figsize=(8, 10)).clf()
    # s = plt.scatter(x2, y2, cmap='jet', marker='.')
    # plt.colorbar(s)
    # plt.show()

    # Calculate the point density
    # xy = np.vstack([np_ug, np_gr])
    # z = gaussian_kde(xy)(xy)
    # print('A gaussian kde calculated')
    #
    # fig, ax = plt.subplots()
    # ax.scatter(np_ug, np_gr, c=z, s=5, edgecolor='', cmap='jet')
    # plt.scatter(sun['u-g'], sun['g-r'], edgecolors='', facecolors='black')
    # plt.show()

    # #
    # plt.scatter(adr5_good['psfMag_r'] - adr5_good['psfMag_i'],
    #             adr5_good['psfMag_i'] - adr5_good['psfMag_z'], alpha=0.1, s=2)
    # plt.scatter(sun['r-i'], sun['i-z'])
    # plt.xlabel('r-i')
    # plt.ylabel('i-z')
    # plt.show()


def adr5_recover():
    adr5 = pd.read_csv(f'{path}init/adr5c_full.csv')
    print(adr5.info())

    adr5['psfMagErr_u'] = pd.to_numeric(adr5['psfMagErr_u'], errors='coerce')
    adr5['psfMagErr_z'] = pd.to_numeric(adr5['psfMagErr_z'], errors='coerce')
    adr5 = adr5.replace(np.nan, -9999.0, regex=True)

    # for i in range(adr5.__len__()):
    # # for i in range(10):
    #     # print(adr5.iloc[i]['psfMagErr_u'])
    #     try:
    #         float(adr5.iloc[i]['psfMagErr_z'])
    #     except ValueError:
    #         # adr5.ix[i, 'psfMagErr_u'] = -9999.0
    #         adr5.iloc[i, adr5.columns.get_loc('psfMagErr_z')] = -9999.0
    #         print(i, adr5.iloc[i]['objID'], adr5.iloc[i]['psfMagErr_z'])

    # adr5['psfMagErr_u'].astype(float)

    adr5['psfMagErr_u'] = adr5['psfMagErr_u'].round(5)
    adr5['psfMagErr_g'] = adr5['psfMagErr_g'].round(5)
    adr5['psfMagErr_r'] = adr5['psfMagErr_r'].round(5)
    adr5['psfMagErr_i'] = adr5['psfMagErr_i'].round(5)
    adr5['psfMagErr_z'] = adr5['psfMagErr_z'].round(5)
    adr5['psfMag_u'] = adr5['psfMag_u'].round(5)
    adr5['psfMag_g'] = adr5['psfMag_g'].round(5)
    adr5['psfMag_r'] = adr5['psfMag_r'].round(5)
    adr5['psfMag_i'] = adr5['psfMag_i'].round(5)
    adr5['psfMag_z'] = adr5['psfMag_z'].round(5)
    adr5['rowv'] = adr5['rowv'].round(8)
    adr5['colv'] = adr5['colv'].round(8)
    adr5['ang_dist'] = adr5['ang_dist'].round(3)
    adr5['dv_dec'] = adr5['dv_dec'].round(3)
    adr5['dv_ra'] = adr5['dv_ra'].round(3)
    adr5['dv_abs'] = adr5['dv_abs'].round(6)
    adr5['mjd'] = adr5['mjd'].round(6)

    print(adr5.info())

    adr5.to_csv(f'{path}init/adr5ca_full.csv', index=False)


def format_adr(adr5):
    adr5['psfMagErr_u'] = adr5['psfMagErr_u'].round(5)
    adr5['psfMagErr_g'] = adr5['psfMagErr_g'].round(5)
    adr5['psfMagErr_r'] = adr5['psfMagErr_r'].round(5)
    adr5['psfMagErr_i'] = adr5['psfMagErr_i'].round(5)
    adr5['psfMagErr_z'] = adr5['psfMagErr_z'].round(5)
    adr5['psfMag_u'] = adr5['psfMag_u'].round(5)
    adr5['psfMag_g'] = adr5['psfMag_g'].round(5)
    adr5['psfMag_r'] = adr5['psfMag_r'].round(5)
    adr5['psfMag_i'] = adr5['psfMag_i'].round(5)
    adr5['psfMag_z'] = adr5['psfMag_z'].round(5)
    adr5['rowv'] = adr5['rowv'].round(5)
    adr5['colv'] = adr5['colv'].round(5)
    adr5['vel'] = adr5['vel'].round(5)
    # adr5['ang_dist'] = adr5['ang_dist'].round(3)
    # adr5['dv_dec'] = adr5['dv_dec'].round(3)
    # adr5['dv_ra'] = adr5['dv_ra'].round(3)
    # adr5['dv_abs'] = adr5['dv_abs'].round(6)
    # adr5['mjd'] = adr5['mjd'].round(6)

    adr5['offsetRa_u'] = adr5['offsetRa_u'].round(5)
    adr5['offsetRa_g'] = adr5['offsetRa_g'].round(5)
    adr5['offsetRa_r'] = adr5['offsetRa_r'].round(5)
    adr5['offsetRa_i'] = adr5['offsetRa_i'].round(5)
    adr5['offsetRa_z'] = adr5['offsetRa_z'].round(5)
    adr5['offsetDec_u'] = adr5['offsetDec_u'].round(5)
    adr5['offsetDec_g'] = adr5['offsetDec_g'].round(5)
    adr5['offsetDec_r'] = adr5['offsetDec_r'].round(5)
    adr5['offsetDec_i'] = adr5['offsetDec_i'].round(5)
    adr5['offsetDec_z'] = adr5['offsetDec_z'].round(5)
    adr5['R2'] = adr5['R2'].round(2)

    adr5['RA'] = adr5['RA'].round(8)
    adr5['DEC'] = adr5['DEC'].round(8)
    adr5['ra'] = adr5['ra'].round(8)
    adr5['dec'] = adr5['dec'].round(8)

    adr5['RA_rate'] = adr5['RA_rate'].round(8)
    adr5['DEC_rate'] = adr5['DEC_rate'].round(8)
    adr5['x'] = adr5['x'].round(8)
    adr5['y'] = adr5['y'].round(8)
    adr5['z'] = adr5['z'].round(8)
    adr5['vx'] = adr5['vx'].round(8)
    adr5['vy'] = adr5['vy'].round(8)
    adr5['vz'] = adr5['vz'].round(8)

    adr5['posunc'] = adr5['posunc'].round(3)
    adr5['centerdist'] = adr5['centerdist'].round(3)
    adr5['geodist'] = adr5['geodist'].round(8)
    adr5['heliodist'] = adr5['heliodist'].round(8)

    adr5['Number'] = adr5['Number'].astype('Int64')

    return adr5


def adr5_known_clear():
    def clear_by_mag(df):
        # clear by magnitude
        max_mag = 24.0
        min_mag = 10.0
        max_mag_err = 2.0
        adr = adr5.loc[(df['psfMag_u'] > min_mag) & (df['psfMag_u'] < max_mag) &
                       (df['psfMagErr_u'] > 0.0) & (df['psfMagErr_u'] < max_mag_err) &
                       (df['psfMag_g'] > min_mag) & (df['psfMag_g'] < max_mag) &
                       (df['psfMagErr_g'] > 0.0) & (df['psfMagErr_g'] < max_mag_err) &
                       (df['psfMag_r'] > min_mag) & (df['psfMag_r'] < max_mag) &
                       (df['psfMagErr_r'] > 0.0) & (df['psfMagErr_r'] < max_mag_err) &
                       (df['psfMag_i'] > min_mag) & (df['psfMag_i'] < max_mag) &
                       (df['psfMagErr_i'] > 0.0) & (df['psfMagErr_i'] < max_mag_err) &
                       (df['psfMag_z'] > min_mag) & (df['psfMag_z'] < max_mag) &
                       (df['psfMagErr_z'] > 0.0) & (df['psfMagErr_z'] < max_mag_err)]
        print(adr.__len__())
        return adr

    def clear_by_dist(df):
        adr = df.loc[df['ang_dist'] > 5.0]
        print(adr.__len__())
        return adr

    def clear_by_velocity(df):
        adr = df.loc[df['sdss_vel_arcsec_hour'] < 8.0]
        print(adr.__len__())
        return adr

    adr5 = pd.read_csv(f'{path}init/adr5d_known.csv')
    # print(adr5.info())

    # clear by magnitude
    # adr1 = clear_by_mag(adr5)
    # adr2 = clear_by_dist(adr5)
    adr2 = clear_by_velocity(adr5)

    # adr['ug'] = adr['psfMag_u']-adr['psfMag_g']
    # adr.plot.scatter(x='psfMag_r', y='ug')
    # plt.show()
    # adr = adr.loc[(adr['psfMag_g'] - adr['psfMag_r']) > 2.0]
    # print(adr1[['ra', 'dec']])
    print(adr2[['ra', 'dec']])
    # adr2.to_csv('./data/wrong_ug.csv')


def adr5_test_coords():
    adr5 = pd.read_csv(f'{path}init/adr5d_known.csv')
    adr109 = adr5[adr5['run'] == 109]
    print(adr109)

    for i in range(adr109.__len__()):
        fname_r = f'frame-r-000109-{adr109.iloc[i]["camcol"]}-{adr109.iloc[i]["field"]:04d}.fits.bz2'
        hdu_r = fits.open(f'{path}r/109/{fname_r}')
        data = hdu_r[0].data
        plt.imshow(data, origin='lower')
        plt.show()


def catalog_add_fields():
    # adr5_konwn = pd.read_csv(f'{path}init/adr5d_known.csv')
    adr5_konwn = pd.read_csv(f'{path}init/adr5ca_full.csv')
    adr_add = pd.read_csv(f'{path}init/adr5d_Helgy716.csv')
    offset_list = ['offsetRa_u', 'offsetRa_g', 'offsetRa_r', 'offsetRa_i', 'offsetRa_z',
                   'offsetDec_u', 'offsetDec_g', 'offsetDec_r', 'offsetDec_i', 'offsetDec_z']

    l = adr5_konwn.__len__()
    # l = 5
    offset = np.zeros((l, 10), dtype=np.float)
    print(offset.shape)
    for i in range(l):
        a = adr5_konwn.iloc[i]['objID']
        add_series = adr_add.loc[adr_add['objID'] == a]
        if add_series.__len__() != 1:
            print(add_series)
        for j in range(10):
            offset[i, j] = add_series[offset_list[j]]
        if i % 1000 == 0:
            print(i)

    for j in range(10):
        adr5_konwn[offset_list[j]] = offset[:, j]

    print(adr5_konwn.head())
    # adr5_konwn.to_csv(f'{path}init/adr5e_full.csv', index=False)
    # adr5_konwn.to_csv(f'{path}init/adr5e_known.csv', index=False)


def clear2():
    df = pd.read_csv(f'{path}init/adr5e_known.csv')
    print(df.info())
    # r = 0.1
    # dfa = df.loc[(
    #     (df['offsetRa_r']*df['offsetRa_r'] + df['offsetDec_r']*df['offsetDec_r'] > r)
    #            )]
    # print(dfa)
    dfb = df.loc[(
            (df['psfMag_u'] > 0) & (df['psfMag_g'] > 0) & (df['psfMag_r'] > 0) &
            (df['psfMag_i'] > 0) & (df['psfMag_z'] > 0) &
            (df['psfMagErr_u'] < 2.0) & (df['psfMagErr_g'] < 1.0) & (df['psfMagErr_r'] < 0.5) &
            (df['psfMagErr_i'] < 0.5) & (df['psfMagErr_z'] < 1.0) &
            # (df['psfMag_u'] < 24) &
            # (df['psfMag_g'] < 23) & (df['psfMag_i'] < 23) & (df['psfMag_z'] < 22) &
            (np.square(df['offsetRa_g'] / np.cos(df['dec'] / 180.0 * np.pi)) +
             np.square(df['offsetDec_g']) > 0.5 ** 2) &
            (np.square(df['offsetRa_u'] / np.cos(df['dec'] / 180.0 * np.pi)) +
             np.square(df['offsetDec_u']) > 0.25 ** 2) &
            (np.square(df['offsetRa_i'] / np.cos(df['dec'] / 180.0 * np.pi)) +
             np.square(df['offsetDec_i']) > 0.1 ** 2)
        # &
        # (df['psfMag_u'] - df['psfMag_r'] < 4.0) & (df['psfMag_u'] - df['psfMag_r'] > 0.0) &
        # (df['psfMag_g'] - df['psfMag_r'] < 2.0) & (df['psfMag_g'] - df['psfMag_r'] > 0.0) &
        # (df['psfMag_i'] - df['psfMag_r'] < 1.0) & (df['psfMag_i'] - df['psfMag_r'] > -1.0) &
        # (df['psfMag_z'] - df['psfMag_r'] < 1.5) & (df['psfMag_z'] - df['psfMag_r'] > -1.5)
    )]
    print(dfb)

    mag_g = np.ma.array(dfb['psfMag_g'].to_numpy(), mask=np.zeros(dfb.__len__()))
    mag_r = np.ma.array(dfb['psfMag_r'].to_numpy(), mask=np.zeros(dfb.__len__()))
    mag_u = np.ma.array(dfb['psfMag_u'].to_numpy(), mask=np.zeros(dfb.__len__()))
    mag_i = np.ma.array(dfb['psfMag_i'].to_numpy(), mask=np.zeros(dfb.__len__()))
    mag_z = np.ma.array(dfb['psfMag_z'].to_numpy(), mask=np.zeros(dfb.__len__()))
    # plt.scatter(mag_r, mag_g)
    # plt.show()

    fit1 = fitting.LinearLSQFitter()
    or_fit1 = fitting.FittingWithOutlierRemoval(fit1, sigma_clip, niter=5, sigma=10.0)
    line_init = models.Linear1D()
    fitted_line_rg, mask_rg = or_fit1(line_init, mag_r, mag_g)
    mag_r.mask = mask_rg
    mag_g.mask = mask_rg
    mag_u.mask = mask_rg
    mag_i.mask = mask_rg
    mag_z.mask = mask_rg

    xx = np.linspace(mag_r.min(), mag_r.max(), 10)
    isshow = False
    if isshow:
        # filtered_data = np.ma.masked_array(y, mask=mask)
        plt.scatter(x=mag_r.data, y=mag_g.data, alpha=0.5, s=5.0, edgecolors='none')
        plt.scatter(x=mag_r, y=mag_g, alpha=0.5, s=5.0, edgecolors='none')
        plt.plot(xx, fitted_line_rg(xx), color='black')
        # plt.title('Sigma clipped magnitude relation')
        # plt.annotate(f'y = {fitted_line.slope.value:.2f} * x + {fitted_line.intercept.value:.2f}', xy=(15, 23))
        # plt.scatter(x=x, y=filtered_data, color='red', alpha=0.2, s=1.0)
        plt.xlabel('psfMag_r (mag)')
        plt.ylabel('psfMag_g (mag)')
        plt.grid(True)
        # plt.savefig('./figs/mag_rg.png', dpi=150)
        plt.show()

    # mag_r2 = mag_r[~mask_rg].copy()
    # mag_u2 = mag_u[~mask_rg].copy()
    fit2 = fitting.LinearLSQFitter()
    or_fit2 = fitting.FittingWithOutlierRemoval(fit2, sigma_clip, niter=5, sigma=5.0)
    fitted_line_ru, mask_ru = or_fit2(line_init, mag_r, mag_u)
    mag_r.mask = np.logical_or(mag_r.mask, mask_ru)
    mag_g.mask = np.logical_or(mag_r.mask, mask_ru)
    mag_u.mask = np.logical_or(mag_r.mask, mask_ru)
    mag_i.mask = np.logical_or(mag_r.mask, mask_ru)
    mag_z.mask = np.logical_or(mag_r.mask, mask_ru)

    if isshow:
        plt.scatter(x=mag_r.data, y=mag_u.data, alpha=0.5, s=5.0, edgecolors='none')
        plt.scatter(x=mag_r, y=mag_u, alpha=0.5, s=5.0, edgecolors='none')
        plt.plot(xx, fitted_line_ru(xx), color='black')
        plt.xlabel('psfMag_r (mag)')
        plt.ylabel('psfMag_u (mag)')
        plt.grid(True)
        # plt.savefig('./figs/mag_ru.png', dpi=150)
        plt.show()

    fit3 = fitting.LinearLSQFitter()
    or_fit3 = fitting.FittingWithOutlierRemoval(fit3, sigma_clip, niter=5, sigma=10.0)
    fitted_line_ri, mask_ri = or_fit3(line_init, mag_r, mag_i)
    mag_r.mask = np.logical_or(mag_r.mask, mask_ri)
    mag_g.mask = np.logical_or(mag_r.mask, mask_ri)
    mag_u.mask = np.logical_or(mag_r.mask, mask_ri)
    mag_i.mask = np.logical_or(mag_r.mask, mask_ri)
    mag_z.mask = np.logical_or(mag_r.mask, mask_ri)

    if isshow:
        plt.scatter(x=mag_r.data, y=mag_i.data, alpha=0.5, s=5.0, edgecolors='none')
        plt.scatter(x=mag_r, y=mag_i, alpha=0.5, s=5.0, edgecolors='none')
        plt.plot(xx, fitted_line_ri(xx), color='black')
        plt.xlabel('psfMag_r (mag)')
        plt.ylabel('psfMag_i (mag)')
        plt.grid(True)
        # plt.savefig('./figs/mag_ri.png', dpi=150)
        plt.show()

    fit4 = fitting.LinearLSQFitter()
    or_fit4 = fitting.FittingWithOutlierRemoval(fit4, sigma_clip, niter=5, sigma=6.0)
    fitted_line_rz, mask_rz = or_fit4(line_init, mag_r, mag_z)
    mag_r.mask = np.logical_or(mag_r.mask, mask_rz)
    mag_g.mask = np.logical_or(mag_r.mask, mask_rz)
    mag_u.mask = np.logical_or(mag_r.mask, mask_rz)
    mag_i.mask = np.logical_or(mag_r.mask, mask_rz)
    mag_z.mask = np.logical_or(mag_r.mask, mask_rz)

    if isshow:
        plt.scatter(x=mag_r.data, y=mag_z.data, alpha=0.5, s=5.0, edgecolors='none')
        plt.scatter(x=mag_r, y=mag_z, alpha=0.5, s=5.0, edgecolors='none')
        plt.plot(xx, fitted_line_rz(xx), color='black')
        plt.xlabel('psfMag_r (mag)')
        plt.ylabel('psfMag_z (mag)')
        plt.grid(True)
        # plt.savefig('./figs/mag_rz.png', dpi=150)
        plt.show()
    print(len(mag_r.mask == 1))

    mag_g1 = df['psfMag_g'].to_numpy()
    mag_r1 = df['psfMag_r'].to_numpy()
    mag_i1 = df['psfMag_i'].to_numpy()
    mag_z1 = df['psfMag_z'].to_numpy()
    mask = (mag_g1 > 0) & (mag_r1 > 0) & (mag_i1 > 0) & (mag_z1 > 0)
    mag_g1 = mag_g1[mask]
    mag_r1 = mag_r1[mask]
    mag_i1 = mag_i1[mask]
    mag_z1 = mag_z1[mask]

    # plt.scatter(x=mag_g1 - mag_r1, y=mag_i1 - mag_z1, alpha=0.5, s=5.0, edgecolors='none')
    # plt.scatter(x=mag_g.data - mag_r.data,
    #             y=mag_i.data - mag_z.data, alpha=0.5, s=5.0, edgecolors='none')
    # plt.scatter(x=mag_g-mag_r, y=mag_i-mag_z, alpha=0.01, s=5.0, edgecolors='none')
    # plt.xlim(0.2, 1.2)
    # plt.ylim(-1.2, 1.2)
    # cls = pd.read_csv(f'{path}init/class_colors.csv')
    # print(cls)
    #
    # for i in range(cls.__len__()):
    #     plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
    #     plt.annotate(cls.iloc[i]['class'],
    #                  xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01))
    # plt.show()

    dfe = dfb.loc[~mag_r.mask]
    print(dfe)
    dfe = format_adr(dfe)
    dfe.to_csv(f'{path}init/adr5e1_known.csv', index=False)


def clear3():
    df = pd.read_csv(f'{path}init/adr5e_full.csv')

    print(df.info())

    dfb = df.loc[(
            (df['psfMag_u'] > 0) & (df['psfMag_g'] > 0) & (df['psfMag_r'] > 0) &
            (df['psfMag_i'] > 0) & (df['psfMag_z'] > 0) &
            # (df['psfMagErr_u'] < 2.0) & (df['psfMagErr_g'] < 1.0) & (df['psfMagErr_r'] < 0.5) &
            # (df['psfMagErr_i'] < 0.5) & (df['psfMagErr_z'] < 1.0) &
            (df['psfMag_u'] < 24) &
            (df['psfMag_g'] < 23) & (df['psfMag_i'] < 23) & (df['psfMag_z'] < 22) &
            (np.square(df['offsetRa_g']) + np.square(df['offsetDec_g']) > 0.5 ** 2) &
            (np.square(df['offsetRa_u']) + np.square(df['offsetDec_u']) < 0.5 ** 2)

        # (np.square(df['offsetRa_u'] / np.cos(df['dec'] / 180.0 * np.pi)) +
        #  np.square(df['offsetDec_u']) > 0.25 ** 2) &
        # (np.square(df['offsetRa_i'] / np.cos(df['dec'] / 180.0 * np.pi)) +
        #  np.square(df['offsetDec_i']) > 0.1 ** 2)
    )]
    print(dfb[['ra', 'dec']].head(40))
    # dfb.to_csv(f'{path}init/adr5e1_full.csv', index=False)
    # df[['ra', 'dec']].to_csv(f'{path}init/adr5e_radec.csv', index=False)


def show_taxonomic():
    df = pd.read_csv(f'{path}init/adr5e3_known.csv')
    mag_g = np.ma.array(df['psfMag_g'].to_numpy(), mask=np.zeros(df.__len__()))
    mag_r = np.ma.array(df['psfMag_r'].to_numpy(), mask=np.zeros(df.__len__()))
    mag_u = np.ma.array(df['psfMag_u'].to_numpy(), mask=np.zeros(df.__len__()))
    mag_i = np.ma.array(df['psfMag_i'].to_numpy(), mask=np.zeros(df.__len__()))
    mag_z = np.ma.array(df['psfMag_z'].to_numpy(), mask=np.zeros(df.__len__()))

    # plt.scatter(x=mag_g1 - mag_r1, y=mag_i1 - mag_z1, alpha=0.5, s=5.0, edgecolors='none')
    # plt.scatter(x=mag_g.data - mag_r.data,
    #             y=mag_i.data - mag_z.data, alpha=0.5, s=5.0, edgecolors='none')
    plt.scatter(x=mag_g - mag_r, y=mag_i - mag_z, alpha=0.2, s=5.0, edgecolors='none')
    plt.xlim(0.2, 1.2)
    plt.ylim(-1.2, 1.2)
    cls = pd.read_csv(f'{path}init/class_colors.csv')
    print(cls)

    for i in range(cls.__len__()):
        plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
        plt.annotate(cls.iloc[i]['class'],
                     xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01))

    plt.savefig('./data/known_colors.png', dpi=300)
    plt.show()

    x, y = mag_g - mag_r, mag_i - mag_z

    xmin, xmax = 0.2, 1.2
    ymin, ymax = -1.2, 1.2
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    f = np.sqrt(np.reshape(kernel(positions).T, xx.shape))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Gaussian Kernel density estimation')
    plt.show()

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')

    for i in range(cls.__len__()):
        plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'],
                    f[int(cls.iloc[i]['g-r']), int(cls.iloc[i]['i-z'])], marker='.', color='black')
        # plt.annotate(cls.iloc[i]['class'],
        #              xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01))

    # fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.show()


def topcat_inter():
    from astropy.samp import SAMPIntegratedClient
    client = SAMPIntegratedClient()
    # client.connect()
    # df = pd.read_csv(f'{path}init/adr4a.csv')
    # params = {}
    # params["url"] = f'{path}init/adr4a.csv'
    # params["name"] = "Robitaille et al. (2008), Table 3"
    # message = {}
    # message["samp.mtype"] = "table.load.votable"
    # message["samp.params"] = params
    # client.notify_all(message)
    # client.disconnect()
    client.get_registered_clients()
    # print(client.get_metadata('c1'))
    client.disconnect()


def clear_adr4():
    # table = ascii.read(f'{path}init/ADR4.dat')
    # print(table[:5])
    # df = table.to_pandas()
    # df.to_csv(f'{path}init/ADR4.csv')
    df = pd.read_csv(f'./data/panstarrs_adr4.csv')
    # print(df)
    print(df[['ra', 'dec']].head(40))
    df = pd.read_csv(f'./data/panstarrs_adr4.csv')


def clear_panstarrs_known():
    df_pns = pd.read_csv(f'{path}/init/PanSTARRS_2_known.csv')
    df = pd.read_csv(f'{path}/init/adr5e1_known.csv')
    print(df_pns)

    df_pns = df_pns[df_pns['Nd'] > 3]
    # print(df_pns)
    # df_pns = df_pns[~((df_pns['rmag'] > 23) | (df_pns['imag'] > 23))]
    # print(df_pns)
    cond1 = (df_pns['rmag'] is not np.nan) & (df_pns['rmag'] - df_pns['psfMag_r'] < 2.5)
    cond2 = (df_pns['imag'] is not np.nan) & (df_pns['imag'] - df_pns['psfMag_i'] < 2.5)
    cond3 = (df_pns['gmag'] is not np.nan) & (df_pns['gmag'] - df_pns['psfMag_g'] < 2.5)

    cond = np.logical_or(cond1, cond2)
    cond = np.logical_or(cond, cond3)
    print(df_pns.loc[cond][['gmag', 'psfMag_g', 'rmag', 'psfMag_r', 'imag', 'psfMag_i']])
    # print(df_pns.loc[cond][['ra', 'dec']].head(40))
    # df_pns = df_pns[(df_pns['rmag'] > 23) | (df_pns['imag'] > 23))]
    df = df[~df['objID'].isin(df_pns['objID'])].reset_index(drop=True)

    cond4 = (df['ang_dist'] < 2.0)
    df = df[cond4].reset_index(drop=True)

    df = format_adr(df)
    # print(df)
    # df.to_csv(f'{path}/init/adr5e3_known.csv', index=False)


def full_sdss16():
    f_out = open('./data/full_sdss_asteroids.csv', 'wt')
    tar = tarfile.open(f'{path}init/asteroids.tgz', "r:gz")
    members = tar.getmembers()
    count = 0
    for member in members:
        if count & 1000 == 0:
            print(count, member.name)
        if os.path.splitext(member.name)[1] == ".csv":
            f = tar.extractfile(member)
            content = f.read().decode().split('\n')
            # print(len(content))
            if len(content) > 2:
                if count == 0:
                    s = f'{content[0]},fitsname\n'
                    # print(s)
                    f_out.write(s)
                for item in content[1:-1]:
                    s = f'{item},{member.name.split("/")[-1:][0].split(".")[0]}\n'
                    f_out.write(s)
                    # print(s)
                count += 1
            # df = pd.DataFrame(content)
            # print(df)
    f_out.close()


def ps1_math():
    # remove duplicates and prepare
    df = pd.read_csv(f'{path}init/adr5f1_full.csv')
    del df['flags']
    df_ps = pd.read_csv(f'{path}init/adr5f1_full_ps1_match.csv')
    print(df_ps)
    df_ps = df_ps.sort_values(by=['objID'], ascending=False).groupby('objID').first().reset_index()
    print(df_ps)
    # print(df_ps['.info()'])

    df = pd.merge(df, df_ps[['objID', 'RAJ2000', 'DEJ2000', 'Nd', 'gmag', 'e_gmag', 'rmag', 'e_rmag',
                             'imag', 'e_imag', 'zmag', 'e_zmag', 'ymag', 'e_ymag', 'angDist'
                             ]], how='left', on='objID')
    # df = format_adr(df)
    # df['angDist'] = df['angDist'].fillna(-9999.0)
    print(df)

    # df.to_csv(f'{path}/init/adr5f1ps_full.csv', index=False)


def check_sdr5ps():
    df = pd.read_csv(f'{path}init/adr5f1ps_full.csv')
    print(df)

    # as minimum two bands
    cond0 = np.zeros(df.__len__(), dtype=int)
    cond0 += (df['psfMag_r'] > 0).to_numpy()
    cond0 += (df['psfMag_g'] > 0).to_numpy()
    cond0 += (df['psfMag_i'] > 0).to_numpy()
    cond0 += (df['psfMag_z'] > 0).to_numpy()
    cond0 += (df['psfMag_u'] > 0).to_numpy()
    # print(cond0)
    # + (df['psfMag_i'] > 0) + (df['psfMag_z'] > 0)
    # df = df[cond0 > 2]

    # checking on PS1
    # cond1 = ((df['angDist'] >= 0.0) & (df['angDist'] <= 2.0)) | (df['angDist'].isna())
    # cond1 = (df['angDist'] >= 0.0) & (df['angDist'] <= 2.0)
    # df = df[cond1]
    cond11 = ((df['rmag'] > 0) & (df['rmag'] - df['psfMag_r'] < 2.5)) \
             | ((df['gmag'] > 0) & (df['gmag'] - df['psfMag_g'] < 2.5)) \
             | ((df['imag'] > 0) & (df['imag'] - df['psfMag_i'] < 2.5)) \
             | ((df['zmag'] > 0) & (df['zmag'] - df['psfMag_z'] < 2.5))
    print(df[cond11])
    df = df[~cond11]
    del df['RAJ2000']
    del df['DEJ2000']
    del df['Nd']
    del df['gmag']
    del df['e_gmag']
    del df['rmag']
    del df['e_rmag']
    del df['imag']
    del df['e_imag']
    del df['zmag']
    del df['e_zmag']
    del df['ymag']
    del df['e_ymag']
    del df['angDist']

    # df.to_csv(f'{path}/init/adr5f2_full.csv', index=False)


def check_offset():
    df = pd.read_csv(f'{path}init/last/sso_tot3b.csv')
    print(df.__len__())

    # as minimum 2 bands g,r
    # cond2 = (df['psfMag_g'] > 0) | (df['psfMag_z'] > 0) | (df['psfMag_i'] > 0)
    # df = df[cond2]

    # offset

    dra = (df['offsetRa_g'] - df['offsetRa_r']).to_numpy()
    ddec = (df['offsetDec_g'] - df['offsetDec_r']).to_numpy()

    isshow = True
    if isshow:
        plt.figure(figsize=(5, 5))
        k = 10
        dra = (df['offsetRa_g'] - df['offsetRa_r']).to_numpy()[::k]
        ddec = (df['offsetDec_g'] - df['offsetDec_r']).to_numpy()[::k]

        cond3 = (dra ** 2 + ddec ** 2 > 0.59 ** 2)

        # cond1 = ((df['offsetRa_r'] - df['offsetRa_g']) ** 2 + (df['offsetDec_r'] - df['offsetDec_g']) ** 2 < 0.6 ** 2)

        plt.scatter(dra, ddec, s=1, facecolors='red', edgecolors='', label='marked')

        # cond = (dra ** 2) + (ddec ** 2) < (6.1 * 1.5) ** 2
        # dra = dra[cond]
        # ddec = ddec[cond]

        # xy = np.vstack([dra, ddec])
        # z = gaussian_kde(xy)(xy)
        # plt.scatter(dra, ddec, c=np.log(z), cmap='Blues', s=1, edgecolors='',
        #             label='approved')
        plt.scatter(dra[cond3], ddec[cond3], cmap='Blues', s=1, edgecolors='',
                    label='candidates')

        circle1 = plt.Circle((0, 0), 0.55, color='red', fill=False, label='0.05 deg/day')
        circle2 = plt.Circle((0, 0), 6.0, color='green', fill=False, label='0.5 deg/day')
        # fig = plt.gcf()
        ax = plt.gcf().gca()
        ax.add_artist(circle1, label='0.05 deg/day')
        ax.add_artist(circle2)
        plt.xlim(-6.2, 6.2)
        plt.ylim(-6.2, 6.2)
        plt.xlabel(r'$\Delta\alpha cos\delta$, (g-r, arcsec)')
        plt.ylabel(r'$\Delta\delta$, (g-r, arcsec)')
        plt.gca().set_aspect('equal')

        # plt.legend(handles=[plt.plot([], ls="-", color=line.get_color())[0]],
        #            labels=[line.get_label()])

        plt.legend(loc=1, markerscale=4)
        # plt.savefig('./figs/offset_gr.png', dpi=300)
        plt.show()

    # df = df[~cond3]
    # print(df[cond3].__len__())

    #
    # # print(df)
    # # print(df[['ra', 'dec']].head(50))
    #
    # df.to_csv(f'{path}/init/last/sso_tot2a_offset.csv', index=False)


def check_mag():
    df = pd.read_csv(f'{path}init/last/sso_tot3c.csv')
    print(df.__len__())

    isshow = False
    if isshow:
        fig = plt.figure(figsize=(5, 14))
        ax_u = fig.add_subplot(511)
        ax_g = fig.add_subplot(512)
        ax_r = fig.add_subplot(513)
        ax_i = fig.add_subplot(514)
        ax_z = fig.add_subplot(515)
        plt.subplots_adjust(left=0.125, bottom=0.05, right=0.95, top=0.98, wspace=None, hspace=0.001)

        xb_lim, xe_lim = 14, 26
        yb_lim, ye_lim = -0.05, 1.49

    k = 5
    # u band
    um = df['psfMag_u'].to_numpy()
    ume = df['psfMagErr_u'].to_numpy()

    x = np.linspace(0, 17, 100)
    y = np.exp(x)
    max_x = 17
    max_y = np.exp(max_x)
    x = 14.3 + 10 * x / max_x
    y = 1.05 - y / max_y
    x1 = (um - 14.3) * max_x / 10.0
    y1 = (1.05 - ume) * max_y
    cond_u = x1 < np.log(y1)
    um_good = um[cond_u]
    ume_good = ume[cond_u]

    if isshow:
        # xb_lim, xe_lim = 15, 26
        # yb_lim, ye_lim = -0.05, 2.0
        cond1 = (um > xb_lim) & (um < xe_lim) & (ume > yb_lim) & (ume < ye_lim)
        um = um[cond1]
        ume = ume[cond1]

        cond2 = (um_good > xb_lim) & (um_good < xe_lim) & (ume_good > yb_lim) & (ume_good < ye_lim)
        um_good = um_good[cond2]
        ume_good = ume_good[cond2]

        um, ume = um[::k], ume[::k]
        um_good, ume_good = um_good[::k], ume_good[::k]

        if not os.path.isfile('./data/uz.pickle'):
            xy = np.vstack([um, ume])
            z = gaussian_kde(xy)(xy)
            print('Done u1')
            with open('./data/uz.pickle', 'wb') as f:
                pickle.dump(z, f)

            xy_good = np.vstack([um_good, ume_good])
            z_good = gaussian_kde(xy_good)(xy_good)
            print('Done u2')
            with open('./data/uz_good.pickle', 'wb') as f:
                pickle.dump(z_good, f)
        else:
            with open('./data/uz.pickle', 'rb') as f:
                z = pickle.load(f)
            with open('./data/uz_good.pickle', 'rb') as f:
                z_good = pickle.load(f)

        idx = z.argsort()
        um, ume, z = um[idx], ume[idx], z[idx]
        idx_g = z_good.argsort()
        um_good, ume_good, z_good = um_good[idx_g], ume_good[idx_g], z_good[idx_g]

        # ax_u.scatter(um, ume, color='grey', s=1, edgecolors='', label='False')
        ax_u.scatter(um, ume, c=np.log(z), cmap='Greys', s=2, edgecolors='', label='u:False')
        ax_u.scatter(um_good, ume_good, c=np.log(z_good), cmap='Purples', s=2, edgecolors='',
                     label='u:True')
        # xedges, yedges = np.linspace(15, 26, 50), np.linspace(-0.05, 2.0, 50)
        # hist, xedges, yedges = np.histogram2d(um_good, ume_good, (xedges, yedges))
        #
        # xidx = np.clip(np.digitize(um_good, xedges), 0, hist.shape[0] - 1)
        # yidx = np.clip(np.digitize(ume_good, yedges), 0, hist.shape[1] - 1)
        # c = hist[xidx, yidx]
        # idx = c.argsort()
        # um_good, ume_good, z = um[idx], ume[idx], c[idx]
        # old = ax_u.scatter(um_good, ume_good, c=np.log10(c), cmap='Purples', s=2)

        # ax_u.hexbin(um, ume, cmap='jet')

        # ax_u.plot(x[40:], y[40:], '--', color='black', label='limit')
        ax_u.plot(x[40:], y[40:], '--', color='black')
        ax_u.set_xlim(xb_lim, xe_lim)
        ax_u.set_ylim(yb_lim, ye_lim)
        # ax_u.set_xlabel('u, (mag)')
        temp = tic.MaxNLocator(3)
        ax_u.yaxis.set_major_locator(temp)
        ax_u.set_xticklabels(())
        ax_u.tick_params(direction='inout', length=10.0)
        ax_u.title.set_visible(False)
        ax_u.set_ylabel(r'u_${err}$, (mag)')
        legend = ax_u.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Purples(.8))

        # plt.savefig('./figs/mag_u_ue.png', dpi=300)
        # plt.close()
        # plt.show()

    # g band
    gm = df['psfMag_g'].to_numpy()
    gme = df['psfMagErr_g'].to_numpy()

    x = np.linspace(0, 17, 100)
    y = np.exp(x)
    max_x = 17
    max_y = np.exp(max_x)
    x = 14.8 + 10 * x / max_x
    y = 1.05 - y / max_y
    x1 = (gm - 14.8) * max_x / 10.0
    y1 = (1.05 - gme) * max_y
    cond_g = x1 < np.log(y1)
    gm_good = gm[cond_g]
    gme_good = gme[cond_g]

    if isshow:
        # xb_lim, xe_lim = 14, 26
        # yb_lim, ye_lim = -0.05, 2.0
        cond1 = (gm > xb_lim) & (gm < xe_lim) & (gme > yb_lim) & (gme < ye_lim)
        gm = gm[cond1]
        gme = gme[cond1]

        cond2 = (gm_good > xb_lim) & (gm_good < xe_lim) & (gme_good > yb_lim) & (gme_good < ye_lim)
        gm_good = gm_good[cond2]
        gme_good = gme_good[cond2]

        # k = 5
        gm, gme = gm[::k], gme[::k]
        gm_good, gme_good = gm_good[::k], gme_good[::k]

        if not os.path.isfile('./data/gz.pickle'):
            xy = np.vstack([gm, gme])
            z = gaussian_kde(xy)(xy)
            print('Done g1')
            with open('./data/gz.pickle', 'wb') as f:
                pickle.dump(z, f)
            xy_good = np.vstack([gm_good, gme_good])
            z_good = gaussian_kde(xy_good)(xy_good)
            print('Done g2')
            with open('./data/gz_good.pickle', 'wb') as f:
                pickle.dump(z_good, f)
        else:
            with open('./data/gz.pickle', 'rb') as f:
                z = pickle.load(f)
            with open('./data/gz_good.pickle', 'rb') as f:
                z_good = pickle.load(f)

        idx = z.argsort()
        gm, gme, z = gm[idx], gme[idx], z[idx]
        idx_good = z_good.argsort()
        gm_good, gme_good, z_good = gm_good[idx_good], gme_good[idx_good], z_good[idx_good]

        # plt.scatter(gm, gme, s=2)
        ax_g.scatter(gm, gme, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='g:False')
        ax_g.scatter(gm_good, gme_good, c=np.log(z_good), cmap='Greens', s=2,
                     edgecolors='', label='g:True')
        # ax_g.plot(x[40:], y[40:], '--', color='black', label='limit')
        ax_g.plot(x[40:], y[40:], '--', color='black')

        ax_g.set_xlim(xb_lim, xe_lim)
        ax_g.set_ylim(yb_lim, ye_lim)
        # ax_g.set_xlabel('g, (mag)')
        # temp = tic.MaxNLocator(3)
        ax_g.yaxis.set_major_locator(temp)
        ax_g.set_xticklabels(())
        ax_g.tick_params(direction='inout', length=10.0)
        ax_g.set_ylabel(r'g_${err}$, (mag)')
        legend = ax_g.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Greens(.8))

        # plt.savefig('./figs/mag_g_ge.png', dpi=300)
        # plt.close()
        # plt.show()

    # r band
    rm = df['psfMag_r'].to_numpy()
    rme = df['psfMagErr_r'].to_numpy()

    x = np.linspace(0, 17, 100)
    y = np.exp(x)
    max_x = 17
    max_y = np.exp(max_x)
    x = 14.8 + 10 * x / max_x
    y = 1.05 - y / max_y
    x1 = (rm - 14.8) * max_x / 10.0
    y1 = (1.05 - rme) * max_y
    cond_r = x1 < np.log(y1)
    rm_good = rm[cond_r]
    rme_good = rme[cond_r]

    if isshow:
        cond1 = (rm > xb_lim) & (rm < xe_lim) & (rme > yb_lim) & (rme < ye_lim)
        rm = rm[cond1]
        rme = rme[cond1]

        cond2 = (rm_good > xb_lim) & (rm_good < xe_lim) & (rme_good > yb_lim) & (rme_good < ye_lim)
        rm_good = rm_good[cond2]
        rme_good = rme_good[cond2]

        # k = 1
        rm, rme = rm[::k], rme[::k]
        rm_good, rme_good = rm_good[::k], rme_good[::k]

        if not os.path.isfile('./data/rz.pickle'):
            xy = np.vstack([rm, rme])
            z = gaussian_kde(xy)(xy)
            print('Done r1')
            with open('./data/rz.pickle', 'wb') as f:
                pickle.dump(z, f)
            xy_good = np.vstack([rm_good, rme_good])
            z_good = gaussian_kde(xy_good)(xy_good)
            print('Done r2')
            with open('./data/rz_good.pickle', 'wb') as f:
                pickle.dump(z_good, f)
        else:
            with open('./data/rz.pickle', 'rb') as f:
                z = pickle.load(f)
            with open('./data/rz_good.pickle', 'rb') as f:
                z_good = pickle.load(f)

        # xy = np.vstack([rm, rme])
        # z = gaussian_kde(xy)(xy)
        # print('Done r1')
        # xy_good = np.vstack([rm_good, rme_good])
        # z_good = gaussian_kde(xy_good)(xy_good)
        # print('Done r2')

        idx = z.argsort()
        rm, rme, z = rm[idx], rme[idx], z[idx]
        idx_good = z_good.argsort()
        rm_good, rme_good, z_good = rm_good[idx_good], rme_good[idx_good], z_good[idx_good]

        # plt.scatter(rm, rme, s=2)
        ax_r.scatter(rm, rme, c=np.log(z), cmap='Greys', s=2, edgecolors='', label='r:False')
        ax_r.scatter(rm_good, rme_good, c=np.log(z_good), cmap='Reds', s=2,
                     edgecolors='', label='r:True')
        # ax_r.plot(x[40:], y[40:], '--', color='black', label='limit')
        ax_r.plot(x[40:], y[40:], '--', color='black')

        ax_r.set_xlim(xb_lim, xe_lim)
        ax_r.set_ylim(yb_lim, ye_lim)
        # ax_r.set_xlabel('r, (mag)')
        ax_r.yaxis.set_major_locator(temp)
        ax_r.set_xticklabels(())
        ax_r.tick_params(direction='inout', length=10.0)
        ax_r.set_ylabel(r'r_${err}$, (mag)')
        legend = ax_r.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Reds(.8))

        # plt.savefig('./figs/mag_r_re.png', dpi=300)
        # plt.close()
        # plt.show()

    # i band
    im = df['psfMag_i'].to_numpy()
    ime = df['psfMagErr_i'].to_numpy()

    x = np.linspace(0, 17, 100)
    y = np.exp(x)
    max_x = 17
    max_y = np.exp(max_x)
    x = 14.5 + 10 * x / max_x
    y = 1.05 - y / max_y
    x1 = (im - 14.5) * max_x / 10.0
    y1 = (1.05 - ime) * max_y
    cond_i = x1 < np.log(y1)
    im_good = im[cond_i]
    ime_good = ime[cond_i]

    if isshow:
        # xb_lim, xe_lim = 14, 26
        # yb_lim, ye_lim = -0.05, 2.0
        cond1 = (im > xb_lim) & (im < xe_lim) & (ime > yb_lim) & (ime < ye_lim)
        im = im[cond1]
        ime = ime[cond1]

        cond2 = (im_good > xb_lim) & (im_good < xe_lim) & (ime_good > yb_lim) & (ime_good < ye_lim)
        im_good = im_good[cond2]
        ime_good = ime_good[cond2]

        # k = 5
        im, ime = im[::k], ime[::k]
        im_good, ime_good = im_good[::k], ime_good[::k]

        if not os.path.isfile('./data/iz.pickle'):
            xy = np.vstack([im, ime])
            z = gaussian_kde(xy)(xy)
            print('Done i1')
            with open('./data/iz.pickle', 'wb') as f:
                pickle.dump(z, f)
            xy_good = np.vstack([im_good, ime_good])
            z_good = gaussian_kde(xy_good)(xy_good)
            print('Done i2')
            with open('./data/iz_good.pickle', 'wb') as f:
                pickle.dump(z_good, f)
        else:
            with open('./data/iz.pickle', 'rb') as f:
                z = pickle.load(f)
            with open('./data/iz_good.pickle', 'rb') as f:
                z_good = pickle.load(f)

        # xy = np.vstack([im, ime])
        # z = gaussian_kde(xy)(xy)
        # print('Done i1')
        # xy_good = np.vstack([im_good, ime_good])
        # z_good = gaussian_kde(xy_good)(xy_good)
        # print('Done i1')

        idx = z.argsort()
        im, ime, z = im[idx], ime[idx], z[idx]
        idx_good = z_good.argsort()
        im_good, ime_good, z_good = im_good[idx_good], ime_good[idx_good], z_good[idx_good]

        # plt.scatter(im, ime, s=2)
        ax_i.scatter(im, ime, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='i:False')
        ax_i.scatter(im_good, ime_good, c=np.log(z_good), cmap='Oranges', s=2,
                     edgecolors='', label='i:True')
        # ax_i.plot(x[40:], y[40:], '--', color='black', label='limit')
        ax_i.plot(x[40:], y[40:], '--', color='black')

        ax_i.set_xlim(xb_lim, xe_lim)
        ax_i.set_ylim(yb_lim, ye_lim)
        # ax_i.set_xlabel('i, (mag)')
        # temp = tic.MaxNLocator(3)
        ax_i.yaxis.set_major_locator(temp)
        ax_i.set_xticklabels(())
        ax_i.tick_params(direction='inout', length=10.0)
        ax_i.title.set_visible(False)
        ax_i.set_ylabel(r'i_${err}$, (mag)')
        legend = ax_i.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Oranges(.8))

        # plt.xlim(14, 26)
        # plt.ylim(-0.05, 2.0)
        # plt.xlabel('i, (mag)')
        # plt.ylabel(r'i_${err}$, (mag)')
        # plt.legend()
        # plt.savefig('./figs/mag_i_ie.png', dpi=300)
        # plt.close()
        # plt.show()

    # z band
    zm = df['psfMag_z'].to_numpy()
    zme = df['psfMagErr_z'].to_numpy()

    x = np.linspace(0, 17, 100)
    y = np.exp(x)
    max_x = 17
    max_y = np.exp(max_x)
    x = 12.5 + 10 * x / max_x
    y = 1.05 - y / max_y
    x1 = (zm - 12.5) * max_x / 10.0
    y1 = (1.05 - zme) * max_y
    cond_z = x1 < np.log(y1)
    zm_good = zm[cond_z]
    zme_good = zme[cond_z]

    if isshow:
        # xb_lim, xe_lim = 14, 26
        # yb_lim, ye_lim = -0.05, 1.5
        cond1 = (zm > xb_lim) & (zm < xe_lim) & (zme > yb_lim) & (zme < ye_lim)
        zm = zm[cond1]
        zme = zme[cond1]

        cond2 = (zm_good > xb_lim) & (zm_good < xe_lim) & (zme_good > yb_lim) & (zme_good < ye_lim)
        zm_good = zm_good[cond2]
        zme_good = zme_good[cond2]

        # k = 10
        zm, zme = zm[::k], zme[::k]
        zm_good, zme_good = zm_good[::k], zme_good[::k]

        if not os.path.isfile('./data/zz.pickle'):
            xy = np.vstack([zm, zme])
            z = gaussian_kde(xy)(xy)
            print('Done z1')
            with open('./data/zz.pickle', 'wb') as f:
                pickle.dump(z, f)
            xy_good = np.vstack([zm_good, zme_good])
            z_good = gaussian_kde(xy_good)(xy_good)
            print('Done z2')
            with open('./data/zz_good.pickle', 'wb') as f:
                pickle.dump(z_good, f)
        else:
            with open('./data/zz.pickle', 'rb') as f:
                z = pickle.load(f)
            with open('./data/zz_good.pickle', 'rb') as f:
                z_good = pickle.load(f)

        # xy = np.vstack([zm, zme])
        # z = gaussian_kde(xy)(xy)
        # print('Done z1')
        # xy_good = np.vstack([zm_good, zme_good])
        # z_good = gaussian_kde(xy_good)(xy_good)
        # print('Done z2')

        idx = z.argsort()
        zm, zme, z = zm[idx], zme[idx], z[idx]
        idx_good = z_good.argsort()
        zm_good, zme_good, z_good = zm_good[idx_good], zme_good[idx_good], z_good[idx_good]

        # plt.scatter(zm, zme, s=2, alpha=0.1, edgecolors='')
        ax_z.scatter(zm, zme, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='z:False')
        ax_z.scatter(zm_good, zme_good, c=np.log(z_good), cmap='Blues', s=2,
                     edgecolors='', label='z:True')
        # ax_z.plot(x[50:], y[50:], '--', color='black', label='limit')
        ax_z.plot(x[50:], y[50:], '--', color='black')

        ax_z.set_xlim(xb_lim, xe_lim)
        ax_z.set_ylim(yb_lim, ye_lim)
        ax_z.set_xlabel('mag')
        # temp = tic.MaxNLocator(3)
        ax_z.yaxis.set_major_locator(temp)
        ax_z.tick_params(direction='inout', length=10.0)
        # ax_z.set_xticklabels(())
        # ax_z.title.set_visible(False)
        ax_z.set_ylabel(r'z_${err}$, (mag)')
        legend = ax_z.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Blues(.8))

        # plt.xlim(13, 24)
        # plt.ylim(-0.05, 2.0)
        # plt.xlabel('z, (mag)')
        # plt.ylabel(r'z_${err}$, (mag)')
        # plt.legend()
        # plt.savefig('./figs/mag_z_ze.png', dpi=300)
        # plt.close()

    # plt.savefig('./figs/mag_err.png', dpi=300)
    plt.show()

    # one = np.ones(df.__len__(), dtype=bool)
    # cond_u = (df['psfMag_u'] < 24.0) & (df['psfMag_u'] > 0.0) & (df['psfMagErr_u'] < 1.0)
    cond_offset = ((df['offsetRa_r'] - df['offsetRa_g']) ** 2 + (
            df['offsetDec_r'] - df['offsetDec_g']) ** 2 < 0.6 ** 2) | \
                  ((df['offsetRa_r'] - df['offsetRa_g']) ** 2 +
                   (df['offsetDec_r'] - df['offsetDec_g']) ** 2 > 100 ** 2)
    df['bo'] = ~cond_offset

    df['bu'] = cond_u
    # cond_g = (df['psfMag_g'] < 24.0) & (df['psfMag_g'] > 0.0) & (df['psfMagErr_g'] < 1.0)
    df['bg'] = cond_g
    # cond_r = (df['psfMag_r'] < 24.0) & (df['psfMag_r'] > 0.0) & (df['psfMagErr_r'] < 1.0)
    df['br'] = cond_r
    # cond_i = (df['psfMag_i'] < 24.0) & (df['psfMag_i'] > 0.0) & (df['psfMagErr_i'] < 1.0)
    df['bi'] = cond_i
    # cond_z = (df['psfMag_z'] < 22.5) & (df['psfMag_z'] > 0.0) & (df['psfMagErr_z'] < 1.0)
    df['bz'] = cond_z
    print(df[~cond_u].__len__(), df[~cond_g].__len__(), df[~cond_r].__len__(),
          df[~cond_i].__len__(), df[~cond_z].__len__(), df[cond_offset].__len__())
    print(df[['bu', 'bg', 'br', 'bi', 'bz', 'bo']])

    df.to_csv(f'{path}/init/last/sso_tot3d.csv', index=False)
    # check the number of bad photometry objects
    # print(df.__len__() - df[df['bz']].__len__())


def check_mag2():
    df = pd.read_csv(f'{path}init/last/sso_tot3b.csv')
    print(df.__len__())

    isshow = False

    k = 10
    # u band
    um = df['psfMag_u'].to_numpy()
    ume = df['psfMagErr_u'].to_numpy()

    u_lim = 24.0
    ue_lim = 1.08
    cond_u = (um < u_lim) & (ume < ue_lim)
    um_good = um[cond_u]
    ume_good = ume[cond_u]

    if not isshow:
        fig = plt.figure()
        ax_u = fig.add_subplot(221)
        ax_g = fig.add_subplot(222)
        ax_i = fig.add_subplot(223)
        ax_z = fig.add_subplot(224)

    if not isshow:
        # cond = (um > 14) & (um < 27) & (ume > 0) & (ume < 3)
        # um = um[cond]
        # ume = ume[cond]

        um, ume = um[::k], ume[::k]
        um_good, ume_good = um_good[::k], ume_good[::k]
        # xy = np.vstack([um, ume])
        # z = gaussian_kde(xy)(xy)

        # xy_good = np.vstack([um_good, ume_good])
        # z_good = gaussian_kde(xy_good)(xy_good)

        # idx = z.argsort()
        # um, ume, z = um[idx], ume[idx], z[idx]
        # idx_g = z_good.argsort()
        # um_good, ume_good, z_good = um_good[idx_g], ume_good[idx_g], z_good[idx_g]

        ax_u.scatter(um, ume, cmap='Greys', s=1, edgecolors='', label='marked')
        # plt.scatter(um, ume, c=np.log(z), cmap='Greys', s=2, edgecolors='', label='outlier')
        # plt.scatter(um_good, ume_good, c=np.log(z_good), cmap='Purples', s=10, edgecolors='',
        #             label='good')
        ax_u.plot([15, u_lim], [ue_lim, ue_lim], '--', color='black', label='edge')
        ax_u.plot([u_lim, u_lim], [ue_lim, 0.0], '--', color='black', label='edge')
        ax_u.set_xlim(15, 26)
        ax_u.set_ylim(-0.05, 2.0)
        ax_u.set_xlabel('u, (mag)')
        ax_u.set_ylabel(r'u_${err}$, (mag)')
        ax_u.legend()
        # plt.savefig('./figs/mag_u_ue.png', dpi=300)
        # plt.close()
        plt.show()

    # # g band
    # gm = df['psfMag_g'].to_numpy()
    # gme = df['psfMagErr_g'].to_numpy()
    #
    #
    # gm_good = gm[cond_g]
    # gme_good = gme[cond_g]
    #
    # if isshow:
    #     cond = (gm > 14) & (gm < 27) & (gme > 0) & (gme < 3)
    #     gm = gm[cond]
    #     gme = gme[cond]
    #
    #     # k = 5
    #     gm, gme = gm[::k], gme[::k]
    #     gm_good, gme_good = gm_good[::k], gme_good[::k]
    #     xy = np.vstack([gm, gme])
    #     z = gaussian_kde(xy)(xy)
    #     xy_good = np.vstack([gm_good, gme_good])
    #     z_good = gaussian_kde(xy_good)(xy_good)
    #
    #     idx = z.argsort()
    #     gm, gme, z = gm[idx], gme[idx], z[idx]
    #     idx_good = z_good.argsort()
    #     gm_good, gme_good, z_good = gm_good[idx_good], gme_good[idx_good], z_good[idx_good]
    #
    #     # plt.scatter(gm, gme, s=2)
    #     plt.scatter(gm, gme, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='outlier')
    #     plt.scatter(gm_good, gme_good, c=np.log(z_good), cmap='Greens', s=10,
    #                 edgecolors='', label='good')
    #     plt.plot(x[20:], y[20:], '--', color='black', label='edge')
    #     plt.xlim(14, 26)
    #     plt.ylim(-0.05, 2.0)
    #     plt.xlabel('g, (mag)')
    #     plt.ylabel(r'g_${err}$, (mag)')
    #     plt.legend()
    #     plt.savefig('./figs/mag_g_ge.png', dpi=300)
    #     plt.close()
    #     # plt.show()
    #
    # # r band
    # rm = df['psfMag_r'].to_numpy()
    # rme = df['psfMagErr_r'].to_numpy()
    #
    # x = np.linspace(0, 17, 100)
    # y = np.exp(x)
    # max_x = 17
    # max_y = np.exp(max_x)
    # x = 14.8 + 10 * x / max_x
    # y = 1.05 - y / max_y
    # x1 = (rm - 14.8) * max_x / 10.0
    # y1 = (1.05 - rme) * max_y
    # cond_r = x1 < np.log(y1)
    # rm_good = rm[cond_r]
    # rme_good = rme[cond_r]
    #
    # if isshow:
    #     cond = (rm > 14) & (rm < 27) & (rme > 0) & (rme < 3)
    #     rm = rm[cond]
    #     rme = rme[cond]
    #
    #     # k = 1
    #     rm, rme = rm[::k], rme[::k]
    #     rm_good, rme_good = rm_good[::k], rme_good[::k]
    #     xy = np.vstack([rm, rme])
    #     z = gaussian_kde(xy)(xy)
    #     print('Kernel 1 complete')
    #     xy_good = np.vstack([rm_good, rme_good])
    #     z_good = gaussian_kde(xy_good)(xy_good)
    #
    #     idx = z.argsort()
    #     rm, rme, z = rm[idx], rme[idx], z[idx]
    #     idx_good = z_good.argsort()
    #     rm_good, rme_good, z_good = rm_good[idx_good], rme_good[idx_good], z_good[idx_good]
    #
    #     # plt.scatter(rm, rme, s=2)
    #     plt.scatter(rm, rme, c=np.log(z), cmap='Greys', s=2, edgecolors='', label='outlier')
    #     plt.scatter(rm_good, rme_good, c=np.log(z_good), cmap='Reds', s=10,
    #                 edgecolors='', label='good')
    #     plt.plot(x[0:], y[0:], '--', color='black', label='edge')
    #     plt.xlim(14, 21.6)
    #     plt.ylim(-0.05, 2.0)
    #     plt.xlabel('r, (mag)')
    #     plt.ylabel(r'r$_{err}$, (mag)')
    #     plt.legend()
    #     plt.savefig('./figs/mag_r_re.png', dpi=300)
    #     plt.close()
    #     # plt.show()
    #
    # # i band
    # im = df['psfMag_i'].to_numpy()
    # ime = df['psfMagErr_i'].to_numpy()
    #
    # x = np.linspace(0, 17, 100)
    # y = np.exp(x)
    # max_x = 17
    # max_y = np.exp(max_x)
    # x = 14.5 + 10 * x / max_x
    # y = 1.05 - y / max_y
    # x1 = (im - 14.5) * max_x / 10.0
    # y1 = (1.05 - ime) * max_y
    # cond_i = x1 < np.log(y1)
    # im_good = im[cond_i]
    # ime_good = ime[cond_i]
    #
    # if isshow:
    #     cond = (im > 14) & (im < 27) & (ime > 0) & (ime < 3)
    #     im = im[cond]
    #     ime = ime[cond]
    #
    #     # k = 5
    #     im, ime = im[::k], ime[::k]
    #     im_good, ime_good = im_good[::k], ime_good[::k]
    #
    #     xy = np.vstack([im, ime])
    #     z = gaussian_kde(xy)(xy)
    #     xy_good = np.vstack([im_good, ime_good])
    #     z_good = gaussian_kde(xy_good)(xy_good)
    #
    #     idx = z.argsort()
    #     im, ime, z = im[idx], ime[idx], z[idx]
    #     idx_good = z_good.argsort()
    #     im_good, ime_good, z_good = im_good[idx_good], ime_good[idx_good], z_good[idx_good]
    #
    #     # plt.scatter(im, ime, s=2)
    #     plt.scatter(im, ime, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='outlier')
    #     plt.scatter(im_good, ime_good, c=np.log(z_good), cmap='Oranges', s=10,
    #                 edgecolors='', label='good')
    #     plt.plot(x[10:], y[10:], '--', color='black', label='edge')
    #     plt.xlim(14, 26)
    #     plt.ylim(-0.05, 2.0)
    #     plt.xlabel('i, (mag)')
    #     plt.ylabel(r'i_${err}$, (mag)')
    #
    #     plt.legend()
    #     plt.savefig('./figs/mag_i_ie.png', dpi=300)
    #     plt.close()
    # plt.show()

    # # z band
    # zm = df['psfMag_z'].to_numpy()
    # zme = df['psfMagErr_z'].to_numpy()
    #
    # x = np.linspace(0, 17, 100)
    # y = np.exp(x)
    # max_x = 17
    # max_y = np.exp(max_x)
    # x = 12.5 + 10 * x / max_x
    # y = 1.05 - y / max_y
    # x1 = (zm - 12.5) * max_x / 10.0
    # y1 = (1.05 - zme) * max_y
    # cond_z = x1 < np.log(y1)
    # zm_good = zm[cond_z]
    # zme_good = zme[cond_z]
    #
    # if isshow:
    #     cond = (zm > 14) & (zm < 27) & (zme > 0) & (zme < 3)
    #     zm = zm[cond]
    #     zme = zme[cond]
    #
    #     # k = 10
    #     zm, zme = zm[::k], zme[::k]
    #     zm_good, zme_good = zm_good[::k], zme_good[::k]
    #
    #     xy = np.vstack([zm, zme])
    #     z = gaussian_kde(xy)(xy)
    #     xy_good = np.vstack([zm_good, zme_good])
    #     z_good = gaussian_kde(xy_good)(xy_good)
    #
    #     idx = z.argsort()
    #     zm, zme, z = zm[idx], zme[idx], z[idx]
    #     idx_good = z_good.argsort()
    #     zm_good, zme_good, z_good = zm_good[idx_good], zme_good[idx_good], z_good[idx_good]
    #
    #     # plt.scatter(zm, zme, s=2, alpha=0.1, edgecolors='')
    #     plt.scatter(zm, zme, c=np.log10(z), cmap='Greys', s=2, edgecolors='', label='outlier')
    #     plt.scatter(zm_good, zme_good, c=np.log(z_good), cmap='Blues', s=10,
    #                 edgecolors='', label='good')
    #     plt.plot(x[10:], y[10:], '--', color='black', label='edge')
    #     plt.xlim(13, 24)
    #     plt.ylim(-0.05, 2.0)
    #     plt.xlabel('z, (mag)')
    #     plt.ylabel(r'z_${err}$, (mag)')
    #     plt.legend()
    #     plt.savefig('./figs/mag_z_ze.png', dpi=300)
    #     plt.close()
    #     # plt.show()


def check_color():
    def seq(x, y, cut_sig, ax, isshow=False):
        sn = 20
        sx = np.linspace(x.min(), x.max(), sn + 1)
        # print(x.min(), x.max())
        h2 = (sx[1] - sx[0]) / 2

        med = np.zeros(sn, dtype=float)
        sig = np.zeros(sn, dtype=float)
        sig2 = np.zeros(sn, dtype=float)

        sy_data = []
        sx_data = []
        # idx = np.argsort(x)

        for i in range(sn):
            cond = (x >= sx[i]) & (x <= sx[i + 1])
            sy_data.append(y[cond])
            sx_data.append(x[cond])
            # print(len(sx_data[i]))

        for i, sy_seq in enumerate(sy_data[:]):
            med[i] = np.ma.median(np.array(sy_seq))
            sig[i] = np.ma.std(sy_seq)
        # print(med)
        # print(sig)

        sy_filtered_data = []
        maxiter = 10
        for i, sy_seq in enumerate(sy_data[:]):
            # print(sy_seq)
            clipped = sigma_clip(sy_seq, sigma=cut_sig, maxiters=maxiter, cenfunc=np.ma.median)
            # clipped, low, upp = stats.sigmaclip(sy_seq, 4, 4)
            # print(clipped)
            sy_filtered_data.append(clipped)
            sig2[i] = np.ma.std(clipped)

            # plt.hist(sy_data[i], alpha=0.5)
            # plt.hist(clipped, alpha=0.5)
            # plt.show()
        # print(sig2)

        # print(sy_filtered_data[0])
        sy_filtered = np.ma.array(sy_filtered_data[0])
        sx_filtered = np.ma.array(sx_data[0])
        for i in range(1, sn):
            sy_filtered = np.ma.concatenate((sy_filtered, sy_filtered_data[i]), axis=None)
            sx_filtered = np.ma.concatenate((sx_filtered, sx_data[i]), axis=None)

        if isshow:
            ax.plot(sx[:-1] + h2, med, 'k:')
            ax.errorbar(sx[:-1] + h2, med, sig2 * cut_sig, fmt='k.', markersize=2,
                        label=rf'{cut_sig}$\sigma$')
            ax.errorbar(sx[:-1] + h2, med, sig2 * cut_sig, fmt='k.', markersize=10)

        return sx_filtered, sy_filtered

    def pltcolor(filter1, filter2, band1, band2, ax):
        band_name = bands[band]
        if isshow:
            if issave:
                cond1 = (filter1 > 14) & (filter1 < 22.5) & (filter2 > -3) & (filter2 < 6)
                filter1 = filter1[cond1]
                filter2 = filter2[cond1]
                cond2 = (band1 > 14) & (band1 < 22.5) & (band2 > -3) & (band2 < 6)
                band1 = band1[cond2]
                band2 = band2[cond2]

                filter1 = filter1[::k]
                filter2 = filter2[::k]
                band1 = band1[::k]
                band2 = band2[::k]
                if not os.path.isfile(f'./data/r-{band_name}.pickle'):
                    xy = np.vstack([band1, band2])
                    z12 = gaussian_kde(xy)(xy)
                    print(f'Done {band_name}.')
                    with open(f'./data/r-{band_name}.pickle', 'wb') as f:
                        pickle.dump(z12, f)
                    xy_good = np.vstack([filter1, filter2])
                    z12_good = gaussian_kde(xy_good)(xy_good)
                    print(f'Done {band_name} good.')
                    with open(f'./data/r-{band_name}_good.pickle', 'wb') as f:
                        pickle.dump(z12_good, f)
                else:
                    with open(f'./data/r-{band_name}.pickle', 'rb') as f:
                        z12 = pickle.load(f)
                    with open(f'./data/r-{band_name}_good.pickle', 'rb') as f:
                        z12_good = pickle.load(f)

                idx = z12.argsort()
                band1, band2, z12 = band1[idx], band2[idx], z12[idx]
                idx_good = z12_good.argsort()
                filter1, filter2, z12_good = filter1[idx_good], filter2[idx_good], z12_good[idx_good]

            if issave:
                ax.scatter(band1, band2, s=2, c=np.log(z12), edgecolors='',
                           cmap='Greys', label=f'{band_name}:False')
                ax.scatter(filter1, filter2, s=2, c=np.log(z12_good), edgecolors='',
                           cmap=cmaps[band], label=f'{band_name}:True')
            else:
                ax.scatter(band1, band2, s=2, alpha=1, color='gray', edgecolors='',
                           label=f'{band_name}: False')
                ax.scatter(filter1, filter2, s=2, color='cyan', edgecolors='',
                           label=f'{band_name}:True')

            ax.set_ylabel(f'{band_name}-r, (mag)')
            ax.set_xlim(14.0, 22.5)
            temp = tic.MaxNLocator(9)
            ax.yaxis.set_major_locator(temp)
            fig.canvas.draw()
            labels_y = [item for item in ax.get_yticklabels()]
            ax.set_yticklabels(labels_y[:-1])
            labels_x = [item for item in ax.get_xticklabels()]
            labels_x[0] = ''
            labels_x[-1] = ''
            ax.set_xticklabels(labels_x)

            ax.tick_params(direction='inout', length=10.0)
            ax.title.set_visible(False)

    df = pd.read_csv(f'{path}init/last/sso_tot3d.csv')
    df = df.sort_values('psfMag_r')
    print(df.__len__())
    k = 5
    isshow = True
    issave = True
    sun = {'u-g': 1.40, 'g-r': 0.45, 'r-i': 0.12, 'i-z': 0.04}

    # if isshow:
    fig = plt.figure(figsize=(5, 14))
    ax_u = fig.add_subplot(411)
    ax_g = fig.add_subplot(412)
    ax_i = fig.add_subplot(413)
    ax_z = fig.add_subplot(414)
    plt.subplots_adjust(left=0.125, bottom=0.05, right=0.95, top=0.98, wspace=None, hspace=0.001)
    bands = ['u', 'g', 'i', 'z']
    cmaps = ['Purples', 'Greens', 'Reds', 'Blues']
    zero_mask = np.zeros(df.__len__(), dtype=bool)

    # r = np.ma.array(data=df['psfMag_r'].to_numpy(), mask=~df['br'].to_numpy())
    r = np.ma.array(data=df['psfMag_r'].to_numpy(), mask=zero_mask)

    # # U band
    # band = 0
    # # ur = np.ma.array(data=(df['psfMag_u'] - df['psfMag_r']).to_numpy(), mask=~df['bu'].to_numpy())
    # ur = np.ma.array(data=(df['psfMag_u'] - df['psfMag_r']).to_numpy(), mask=zero_mask)
    # r_filtered, u_filtered = seq(r, ur, 3, ax_u, isshow=True)
    # df['bru'] = ~u_filtered.mask
    # print('U', df[df['bru'] == True].__len__())
    # if isshow:
    #     ax_u.set_ylim(-2, 6)
    #     pltcolor(r_filtered, u_filtered, r, ur, ax_u)
    #     legend = ax_u.legend(loc=2, markerscale=4)
    #     legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    #     legend.legendHandles[1].set_color(plt.cm.Purples(.8))

    # # G
    # # mask = (df['psfMag_g'] - df['psfMag_r'] < 4) | (df['bg'].all())
    # mask = df['bg']
    # # gr = np.ma.array(data=(df['psfMag_g'] - df['psfMag_r']).to_numpy(), mask=~mask.to_numpy())
    # gr = np.ma.array(data=(df['psfMag_g'] - df['psfMag_r']).to_numpy(), mask=zero_mask)
    # r_filtered, g_filtered = seq(r, gr, 5, ax_g, isshow=True)
    # df['brg'] = ~g_filtered.mask
    # print('G', df[df['brg'] == True].__len__())
    # band = 1
    # if isshow:
    #     ax_g.set_ylim(-2, 6)
    #     pltcolor(r_filtered, g_filtered, r, gr, ax_g)
    #     legend = ax_g.legend(loc=2, markerscale=4)
    #     legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    #     legend.legendHandles[1].set_color(plt.cm.Greens(.8))

    # # I band
    # # ir = np.ma.array(data=(df['psfMag_i'] - df['psfMag_r']).to_numpy(), mask=~df['bi'].to_numpy())
    # ir = np.ma.array(data=(df['psfMag_i'] - df['psfMag_r']).to_numpy(), mask=zero_mask)
    # r_filtered, i_filtered = seq(r, ir, 3, ax_i, isshow=True)
    # df['bri'] = ~i_filtered.mask
    # print('I', df[df['bri'] == True].__len__())
    # band = 2
    # if isshow:
    #     ax_i.set_ylim(-3, 5)
    #     pltcolor(r_filtered, i_filtered, r, ir, ax_i)
    #     legend = ax_i.legend(loc=2, markerscale=4)
    #     legend.legendHandles[0].set_color(plt.cm.Greys(.8))
    #     legend.legendHandles[1].set_color(plt.cm.Oranges(.8))

    # Z band
    # zr = np.ma.array(data=(df['psfMag_z'] - df['psfMag_r']).to_numpy(), mask=~df['bz'].to_numpy())
    zr = np.ma.array(data=(df['psfMag_z'] - df['psfMag_r']).to_numpy(), mask=zero_mask)
    r_filtered, z_filtered = seq(r, zr, 3, ax_z, isshow=True)
    df['brz'] = ~z_filtered.mask
    print('Z', df[df['brz'] == True].__len__())
    band = 3
    if isshow:
        ax_z.set_ylim(-3, 5)
        pltcolor(r_filtered, z_filtered, r, zr, ax_z)
        ax_z.set_xlabel('r, (mag)')
        legend = ax_z.legend(loc=2, markerscale=4)
        legend.legendHandles[0].set_color(plt.cm.Greys(.8))
        legend.legendHandles[1].set_color(plt.cm.Blues(.8))

    # # plt.savefig('./figs/color_cond.png', dpi=300)
    # plt.show()

    # df = df.sort_index()
    # print(df[['bru', 'brg', 'bri', 'brz', 'bu', 'bg', 'br', 'bi', 'bz']].tail(40))
    # df = format_adr(df)
    # df.to_csv(f'{path}init/last/sso_tot3ez1.csv', index=False)


def check_known():
    df = pd.read_csv(f'{path}init/adr5f5_full.csv')

    Table(names=('a', 'b', 'c'), dtype=('f4', 'i4', 'S2'))

    skybot = Skybot()
    for j in np.arange(0, df.__len__()):
        # for j in np.arange(0, 10000):
        t1 = Time(df.iloc[j]['TAIr'], format='mjd', scale='tai')
        time_obs = Time(t1.tai, format='mjd', scale='utc')
        coords = SkyCoord(ra=df.iloc[j]['ra'], dec=df.iloc[j]['dec'], unit="deg")
        # print(coords.to_string())
        # print(time_obs.iso)

        try:
            objects = skybot.cone_search(coords, 30 * u.arcsec, time_obs.iso, location='645',
                                         get_raw_response=False, cache=True)
        except:
            print(f'{j} It was error load error')
            out_table.add_row(unknown_table[0])
        else:
            print(f'{j} loaded {len(objects)} asteroid')
            # print(objects.dtype)
            # if num_asteroids > 0:
            #     adr5.loc[i, 'name'] = objects[3].split('|')[1]
            #     adr5.loc[i, 'ra_skybot'] = objects[3].split('|')[2]
            #     adr5.loc[i, 'dec_skybot'] = objects[3].split('|')[3]
            #     # coord = SkyCoord(ra_know, dec_know, unit=(u.hourangle, u.deg))
            #
            #     adr5.loc[i, 'V'] = objects[3].split('|')[5]
            #     adr5.loc[i, 'ang_dist'] = objects[3].split('|')[7]
            #     adr5.loc[i, 'dv_ra'] = objects[3].split('|')[8]  # arcsec/h
            #     adr5.loc[i, 'dv_dec'] = objects[3].split('|')[9]  # arcsec/h
            #     adr5.loc[i, 'dv_abs'] = np.sqrt(
            #         float(adr5.loc[i, 'dv_ra']) ** 2 + float(adr5.loc[i, 'dv_dec']) ** 2)
            #     adr5.loc[i, 'iso'] = Time(adr5.loc[i, 'mjd_r'], format='mjd').iso
            #     adr5.loc[i, 'sdss_coords'] = SkyCoord(ra=adr5.loc[i, 'ra'], dec=adr5.loc[i, 'dec'],
            #                                           unit="deg").to_string('hmsdms')
            #     adr5.loc[i, 'sdss_vel_arcsec_hour'] = adr5.loc[i, 'velocity'] * 3600.0 / 24.0
            # elif num_asteroids == 0:
            #     adr5.loc[i, 'name'] = 'unknown'
            # else:
            #     count_request += 1
        time.sleep(0.1)
        if j == 0:
            out_table = objects
            unknown_table = objects.copy()
            unknown_table[0] = (0.0, 'unknown', 0.0, 0.0, 'unknown', 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            print(unknown_table)
            # for i, item in enumerate(unknown_table):
            #     unknown_table[item] = skybot_unknown[i]
        else:
            for asteroid in objects:
                out_table.add_row(asteroid)
    # ascii.write(out_table, f'{path}init/adr5f6_full.csv', format='csv')


def adr4_obj_list():
    t = ascii.read(f'{path}init/ADR4.dat')
    df = t.to_pandas()
    # df.to_csv(f'{path}init/ADR4.csv', index=False)
    # df_ident = df[['col1', 'col2', 'col3', 'col4', 'col5']]
    # print(df_ident)
    # df_ident.columns = ["moID", "run", "col", "field", "object"]
    # df_ident.to_csv(f'{path}init/ADR4_ident.csv', index=False)

    df_coords = df[['col1', 'col9', 'col10']]
    df_coords.columns = ["moID", "ra", "dec"]
    # df_coords.to_csv(f'{path}init/ADR4_coords.csv', index=False)

    # df = pd.read_table(f'{path}init/ADR4.dat')
    # print(df.__len__())
    # df_ident = df['col1', 'col2', 'col3', 'col4', 'col5']
    # df_ident.columns = ["moID", "run", "col", "field", "object"]
    # print(df_ident)
    # ascii.write(df_ident, f'{path}init/ADR4_ident.dat', format='csv')


def adr4i_compare():
    df_adr4i = pd.read_csv(f'{path}init/adr4i.csv')
    # print(df_adr4i)
    df_ident = pd.read_csv(f'{path}init/ADR4_ident.dat')
    print(df_ident.__len__() - df_adr4i.__len__())
    df1 = df_adr4i[['run', 'camcol', 'field', 'obj']]
    df2 = df_ident[['col1', 'col2', 'col3', 'col4', 'col5', 'col8']]
    # df2.columns = ['moID', 'run', 'camcol', 'field', 'obj', 'mjd']
    # print(df1.info())
    # print(df2.info())

    df = pd.concat([df1, df2]).drop_duplicates(keep=False)
    print(df)

    # df_adr4i_set = set(df_adr4i['run'].to_numpy())
    # df_ident_set = set(df_ident['col2'].to_numpy())
    #
    # print(len(df_adr4i_set))
    # print(len(df_ident_set))
    # adr_diff = df_ident_set - df_adr4i_set
    # print(adr_diff)
    #
    # cond = df_ident['col2'].isin(adr_diff)
    # df = df_ident[cond]

    # df_set = set(df['run'].to_numpy())
    # df_set = sorted(df_set)
    # print(len(df_set))
    # for i, item in enumerate(df_set):
    #     print(i, item, len(df[df['run'] == item]))
    #
    # print(df[df['run'] == 2650])

    # df.to_csv(f'{path}/init/ADR4_ident_diff_.csv', index=False)


def cas_dr7():
    df_adr4 = pd.read_csv(f'{path}init/ADR4.csv')
    df_adr4i = pd.read_csv(f'{path}init/adr4i.csv')

    mask = df_adr4['col1'].isin(df_adr4i['adr4name'])
    df = df_adr4[~mask]
    print(df)
    # df.to_csv(f'{path}/init/ADR4i_diff.csv', index=False)

    df_diff = df[['col1', 'col2', 'col3', 'col4', 'col5', 'col8']]
    df_diff.columns = ['moID', 'run', 'camcol', 'field', 'obj', 'mjd']
    df_diff.to_csv(f'{path}/init/ADR4i_diff2.csv', index=False)

    # pass
    # df = pd.read_csv(f'{path}init/ADR4_coords.csv')
    # df1 = df.iloc[0:1000]
    # # print(df.iloc[0:1000])
    # df1.columns = ['name', 'ra', 'dec']
    # df1.to_csv(f'{path}/init/ADR4_001.csv', index=False)
    # with open(f'{path}/init/adr4_1.txt', 'wt') as f:
    #
    #     for i in range(0, 10):
    #         print(df.iloc[[:10]])
    #         # f.write(df[i]['moID', 'ra', 'dec'])
    # df = pd.read_csv(f'{path}init/adr4i.csv')


def adr4i_merge():
    def tai2mjd(tai):
        t1 = Time(tai, format='mjd', scale='tai')
        t2 = Time(t1.tai, format='mjd', scale='utc')
        return t2

    df_adr4i = pd.read_csv(f'{path}init/adr4i.csv')
    df_strip82 = pd.read_csv(f'{path}init/adr4strip82.csv')

    df_adr4i['TAIr'] = tai2mjd(df_adr4i['TAIr'].to_list())
    df_adr4i.rename(columns={'TAIr': 'mjd_r'}, inplace=True)
    print('r done')
    df_adr4i['TAIu'] = tai2mjd(df_adr4i['TAIu'].to_list())
    df_adr4i.rename(columns={'TAIu': 'mjd_u'}, inplace=True)
    print('u done')
    df_adr4i['TAIg'] = tai2mjd(df_adr4i['TAIg'].to_list())
    df_adr4i.rename(columns={'TAIg': 'mjd_g'}, inplace=True)
    print('g done')
    df_adr4i['TAIi'] = tai2mjd(df_adr4i['TAIi'].to_list())
    df_adr4i.rename(columns={'TAIi': 'mjd_i'}, inplace=True)
    print('i done')
    df_adr4i['TAI_z'] = tai2mjd(df_adr4i['TAI_z'].to_list())
    df_adr4i.rename(columns={'TAI_z': 'mjd_z'}, inplace=True)
    print('z done')

    df = pd.concat([df_adr4i, df_strip82])
    print(df)

    df.to_csv(f'{path}/init/adr4i1.csv', index=False)


def adr4i_adr4bis_merge():
    def tai2mjd(tai):
        t1 = Time(tai, format='mjd', scale='tai')
        t2 = Time(t1.tai, format='mjd', scale='utc')
        return t2

    df_adr4 = pd.read_csv(f'{path}init/adr4i1.csv')
    del df_adr4['adr4name']

    df_adr4bis = pd.read_csv(f'{path}init/adr4bis.csv')
    del df_adr4bis['rerun']
    del df_adr4bis['nDetect']
    df_adr4bis['TAIr'] = tai2mjd(df_adr4bis['TAIr'].to_list())
    df_adr4bis.rename(columns={'TAIr': 'mjd_r'}, inplace=True)
    print('r done')
    df_adr4bis['TAIg'] = tai2mjd(df_adr4bis['TAIg'].to_list())
    df_adr4bis.rename(columns={'TAIg': 'mjd_g'}, inplace=True)
    print('g done')
    df_adr4bis['TAIu'] = tai2mjd(df_adr4bis['TAIu'].to_list())
    df_adr4bis.rename(columns={'TAIu': 'mjd_u'}, inplace=True)
    print('u done')
    df_adr4bis['TAIi'] = tai2mjd(df_adr4bis['TAIi'].to_list())
    df_adr4bis.rename(columns={'TAIi': 'mjd_i'}, inplace=True)
    print('i done')
    df_adr4bis['TAIz'] = tai2mjd(df_adr4bis['TAIz'].to_list())
    df_adr4bis.rename(columns={'TAIz': 'mjd_z'}, inplace=True)
    print('z done')

    df = pd.concat([df_adr4, df_adr4bis])
    print(df)

    df.to_csv(f'{path}/init/adr4j1.csv', index=False)


def sdss_query1():
    df = pd.read_csv(f'{path}/init/full_sdss_asteroids.csv')

    k = 50
    for id, i in enumerate(range(16000, 18000)):
        beg = i * k
        end = (i + 1) * k
        print(id, beg, end)
        df_co = df.iloc[beg:end]
        if df_co.__len__() > 0:
            df_co = df_co[['RA', 'DEC']]
            # print(df_co)
            co = SkyCoord(df_co['RA'].to_numpy(), df_co['DEC'].to_numpy(), frame='icrs', unit='deg')
            # print(co)

            # co = SkyCoord(ra, dec, frame='icrs', unit ='deg')
            result = SDSS.query_region(co,
                                       radius=10.0 * u.arcsec,
                                       photoobj_fields=['objID', 'run', 'rerun', 'camcol', 'field', 'obj', 'ra', 'dec',
                                                        'TAI_u', 'TAI_g', 'TAI_r', 'TAI_i', 'TAI_z',
                                                        'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
                                                        'psfMagErr_u', 'psfMagErr_g', 'psfMagErr_r', 'psfMagErr_i',
                                                        'psfMagErr_z',
                                                        'rowv', 'colv',
                                                        'offsetRa_u', 'offsetDec_u', 'offsetRa_g', 'offsetDec_g',
                                                        'offsetRa_r', 'offsetDec_r', 'offsetRa_i', 'offsetDec_i',
                                                        'offsetRa_z', 'offsetDec_z'],
                                       # field_help=True,
                                       data_release=16)
            # result = SDSS.query_photoobj(run=3427, camcol=2, field=39, data_release=16)
            print(result)
            if result != None:
                if id == 0:
                    df_out = result.to_pandas()
                else:
                    df_out = pd.concat([df_out, result.to_pandas()])
    df_out.to_csv(f'{path}/init/test_sdss_009.csv', index=False)


#     query = """select top 10
# ra, dec, bestObjID
# from PhotoObj
# where
# run=4475 AND rerun=157 AND camcol=3 AND field=158"""
#     query = "select top 10                        z, ra, dec, bestObjID                      from                        photoObj                      where                        class = 'galaxy'                        and z > 0.3                        and zWarning = 0"
#     res = SDSS.query_sql(query)
#     print(res[:5])

# co = SkyCoord('0h8m05.63s +14d50m23.3s')
# result = SDSS.query_region(co, data_release=16)
# print(result[:5])

# df = pd.read_csv(f'{path}/init/window_flist.csv')
#
# # table = hdu[0].data
# # df = pd.DataFrame(table)
# cond = df['SCORE'] > 0.1
# df = df[cond]
# print(df)

# for i in range(689478, 689479):
#     print(df.iloc[i][['RUN', 'RERUN', 'CAMCOL', 'FIELD']])
#     result = SDSS.get_images(run=df.iloc[i]['RUN'], rerun=df.iloc[i]['RERUN'],
#                              camcol=df.iloc[i]['CAMCOL'], field=df.iloc[i]['FIELD'],
#                              # band='gr',
#                              data_release=7)
#     print(result)


def mjd2tai():
    # mjd = [51464.1, 51464.3]
    # du, dg, di, dz = 143, 287, 72, 215
    #
    # t1 = Time(mjd, format='mjd', scale='utc')
    # t2 = Time(t1.tai, format='mjd', scale='tai')
    # t3 = t1.mjd - t2.mjd
    # print(t1.mjd, t2.mjd, t3*24*3600)

    # t1 = Time(tai[idx] - shift, format='mjd', scale='tai')
    # time_obs = Time(t1.tai, format='mjd', scale='utc')
    tai = 4542575283.92
    tai = 4542575292.531009
    mjd = tai / (24 * 3600)
    constantOffset = 2430000
    t1 = Time(mjd, format='mjd', scale='utc')
    print(t1.iso, t1.jd)


def sdss_inside():
    hdu = fits.open(f'{path}/init/window_flist.fits')
    table = hdu[1].data
    id = 0
    r_beg, r_end = table['MU_START'][id], table['MU_END'][id]
    d_beg, d_end = table['NU_START'][id], table['NU_END'][id]

    plt.plot([r_beg, r_end, r_end, r_beg, r_beg], [d_beg, d_beg, d_end, d_end, d_beg])
    plt.show()


def check_neas():
    df = pd.read_csv(f'{path}/init/dr16nea1.csv')
    l = df.__len__()
    df['s_type'] = np.full(l, 'unknown')
    df['name'] = np.full(l, 'unknown')
    df['number'] = np.full(l, '000000')
    # table = Table(names=['ra', 'dec', 'iso_date', 'type', 'name', 'number'])
    skybot = Skybot()
    for j in np.arange(0, df.__len__()):
        # for j in np.arange(0, 10000):
        t1 = Time(df.iloc[j]['TAI_r'], format='mjd', scale='tai')
        time_obs = Time(t1.tai, format='mjd', scale='utc')
        coords = SkyCoord(ra=df.iloc[j]['ra'], dec=df.iloc[j]['dec'], unit="deg")
        # print(coords.to_string())
        # print(time_obs.iso)

        try:
            objects = skybot.cone_search(coords, 30 * u.arcsec, time_obs.iso, location='645',
                                         get_raw_response=False, cache=True)
        except:
            print(j, coords, time_obs.iso)
        else:
            print(objects)
            print(j, coords, time_obs.iso, objects['Type'])
            df.iloc[j, df.columns.get_loc('s_type')] = objects['Type']
            df.iloc[j, df.columns.get_loc('number')] = objects['Number']
            df.iloc[j, df.columns.get_loc('name')] = objects['Name']
    print(df.head())

    df.to_csv(f'{path}/init/dr16nea1s.csv')


def remove_duplicates():
    # df = pd.read_csv(f'{path}/init/last/sso_str82ps2_13.csv')
    # by_groups = df.groupby(["run", 'camcol'])
    # for g in list(by_groups)[:3]:
    #     print(g)
    # df = pd.read_csv(f'{path}init/last/sso_str82ps1.csv')
    # # df = pd.read_csv(f'{path}init/last/sso_dr16ps1.csv')
    # df = df[df['TAI_r'] > 50000]
    # df['TAI_u'] = round(df['TAI_u'] * 24. * 3600., 1)
    # df['TAI_g'] = round(df['TAI_g'] * 24. * 3600., 1)
    # df['TAI_r'] = round(df['TAI_r'] * 24. * 3600., 1)
    # df['TAI_i'] = round(df['TAI_i'] * 24. * 3600., 1)
    # df['TAI_z'] = round(df['TAI_z'] * 24. * 3600., 1)
    # del df['GroupID']
    # del df['GroupSize']
    # # df.to_csv(f'{path}init/last/sso_dr16ps1t.csv', index=False)
    # df.to_csv(f'{path}init/last/sso_str82ps1t.csv', index=False)

    df = pd.read_csv(f'{path}init/last/sso_tot1.csv')
    df['TAI_u'] = round(df['TAI_u'], 1)
    df['TAI_g'] = round(df['TAI_g'], 1)
    df['TAI_r'] = round(df['TAI_r'], 1)
    df['TAI_i'] = round(df['TAI_i'], 1)
    df['TAI_z'] = round(df['TAI_z'], 1)

    # df = df.sort_values(['ra'])
    # df = df[df['TAI_r'] > 50000]
    # df = df.iloc[0:50000]

    # df_non = df[df["GroupID"].isna()]
    # del df_non['GroupID']
    # del df_non['GroupSize']
    # df_non.to_csv(f'{path}/init/last/groups/sso_tot1_non.csv', index=False)

    # dd = df.iloc[0:500]
    df_non = df[df["GroupID"].isna()][:1]
    print(df_non)

    df = df.astype({'objID': np.int64, 'TAI_u': np.int64, "TAI_g": np.int64,
                    "TAI_r": np.int64, "TAI_i": np.int64, "TAI_z": np.int64})
    print(df)
    print('Load complete')

    by_groups = df.groupby(["GroupID"])
    print('Groups complete', by_groups.size())
    b, l = 12, 100000
    for idx2, frame2 in by_groups:
        if int(idx2) % 1000 == 0:
            print(idx2)

        if (idx2 >= b * l) and (idx2 < (b + 1) * l):
            # frame2 = frame2.reset_index()
            time = frame2.iloc[0]['TAI_r']
            frame2['status'] = np.abs(frame2['TAI_r'] - time) < 60
            # print(frame2[['TAI_r', 'status']])
            new_group = frame2.groupby('status')
            for idx, frame in new_group:
                # print(frame[['TAI_r', 'status', 'ra', 'dec']])
                frame = frame.reset_index()

                id_u = frame["psfMag_u"].idxmin()
                id_g = frame["psfMag_g"].idxmin()
                id_r = frame["psfMag_r"].idxmin()
                id_i = frame["psfMag_i"].idxmin()
                id_z = frame["psfMag_z"].idxmin()
                cur_frame = frame['rerun'].idxmax()
                out_frame = pd.DataFrame(frame[cur_frame:cur_frame + 1])
                # print(frame)
                # out_frame.at[0] = frame.iloc[id_r]

                out_frame.at[0, "psfMag_u"] = frame.iloc[id_u]["psfMag_u"]
                out_frame.at[0, "psfMag_g"] = frame.iloc[id_g]["psfMag_g"]
                out_frame.at[0, "psfMag_r"] = frame.iloc[id_r]["psfMag_r"]
                out_frame.at[0, "psfMag_i"] = frame.iloc[id_i]["psfMag_i"]
                out_frame.at[0, "psfMag_z"] = frame.iloc[id_z]["psfMag_z"]

                out_frame.at[0, "psfMagErr_u"] = frame.iloc[id_u]["psfMagErr_u"]
                out_frame.at[0, "psfMagErr_g"] = frame.iloc[id_g]["psfMagErr_g"]
                out_frame.at[0, "psfMagErr_r"] = frame.iloc[id_r]["psfMagErr_r"]
                out_frame.at[0, "psfMagErr_i"] = frame.iloc[id_i]["psfMagErr_i"]
                out_frame.at[0, "psfMagErr_z"] = frame.iloc[id_z]["psfMagErr_z"]

                out_frame.at[0, "offsetRa_u"] = frame.iloc[id_u]["offsetRa_u"]
                out_frame.at[0, "offsetRa_g"] = frame.iloc[id_g]["offsetRa_g"]
                out_frame.at[0, "offsetRa_r"] = frame.iloc[id_r]["offsetRa_r"]
                out_frame.at[0, "offsetRa_i"] = frame.iloc[id_i]["offsetRa_i"]
                out_frame.at[0, "offsetRa_z"] = frame.iloc[id_z]["offsetRa_z"]

                out_frame.at[0, "offsetDec_u"] = frame.iloc[id_u]["offsetDec_u"]
                out_frame.at[0, "offsetDec_g"] = frame.iloc[id_g]["offsetDec_g"]
                out_frame.at[0, "offsetDec_r"] = frame.iloc[id_r]["offsetDec_r"]
                out_frame.at[0, "offsetDec_i"] = frame.iloc[id_i]["offsetDec_i"]
                out_frame.at[0, "offsetDec_z"] = frame.iloc[id_z]["offsetDec_z"]

                # out_frame.at[0, "ra"] = frame.iloc[id_r]["ra"]
                # out_frame.at[0, "dec"] = frame.iloc[id_r]["dec"]
                out_frame.at[0, "type_r"] = frame.iloc[id_r]["type_r"]
                # out_frame.at[0, "objID"] = frame.iloc[id_r]["objID"]
                # out_frame.at[0, "run"] = frame.iloc[id_r]["run"]
                # out_frame.at[0, "rerun"] = frame.iloc[id_r]["rerun"]
                # out_frame.at[0, "camcol"] = frame.iloc[id_r]["camcol"]
                # out_frame.at[0, "field"] = frame.iloc[id_r]["field"]
                # out_frame.at[0, "obj"] = frame.iloc[id_r]["obj"]
                out_frame.at[0, "rowv"] = frame.iloc[id_r]["rowv"]
                out_frame.at[0, "colv"] = frame.iloc[id_r]["colv"]
                out_frame.at[0, "vel"] = frame.iloc[id_r]["vel"]
                # print(out_frame, end='\n\n')

                df_non = df_non.append(out_frame[:1], ignore_index=True)
                del out_frame
            # if int(idx[0]) % 10000 == 0:
            #     gc.collect()
        # else:
        #     break

    del df_non['index']
    del df_non['GroupID']
    del df_non['GroupSize']
    del df_non['status']
    print(df_non)
    df_non.to_csv(f'{path}/init/last/groups/sso_tot1_{b:02d}t.csv', index=False)


def concat_csv():
    df = pd.read_csv(f'{path}init/last/groups/sso_tot1_non.csv')
    for b in range(0, 12):
        print(b)
        dd = pd.read_csv(f'{path}init/last/groups/sso_tot1_{b:02d}t.csv')
        df = pd.concat([df, dd])
    print(df)
    # cond = (df['TAI_r'] - df['TAI_u'] < 1000) & (df['TAI_r'] - df['TAI_g'] < 1000) & \
    #        (df['TAI_r'] - df['TAI_i'] < 1000) & (df['TAI_r'] - df['TAI_z'] < 1000)
    # df = df[cond]
    df.to_csv(f'{path}/init/last/groups/sso_tot2.csv', index=False)


def clean_tai():
    df = pd.read_csv(f'{path}/init/last/sso_tot2.csv')
    cond = (df['TAI_r'] - df['TAI_u'] < 1000) & (df['TAI_r'] - df['TAI_g'] < 1000) & \
           (df['TAI_r'] - df['TAI_i'] < 1000) & (df['TAI_r'] - df['TAI_z'] < 1000)
    df = format_adr(df[cond])
    print(df)
    df.to_csv(f'{path}/init/last/sso_tot2a.csv', index=False)


def skybot_check_tot():
    skybot = Skybot()

    df = pd.read_csv(f'{path}/init/last/sso_tot2c.csv')
    df = df.sort_values('psfMag_r')
    print(df)

    l = 10000
    # dsky = pd.DataFrame(data=df.iloc[b*l:(b+1)*l]['objID'])

    for b in range(126, 127):
        dobj_sky = pd.DataFrame({'objID': []})
        for idx in range(b * l, (b + 1) * l):
            if idx > 1264248:
                break
            # tai to utc
            # ibjID = df.iloc[idx]['objID'].astype(np.int64)
            tai = df.iloc[idx]['TAI_r'] / 3600. / 24.0
            t1 = Time(tai, format='mjd', scale='tai')
            time_obs = Time(t1.tai, format='mjd', scale='utc')
            coords = SkyCoord(ra=df.iloc[idx]['ra'], dec=df.iloc[idx]['dec'], unit="deg")
            iso_coords = coords.ra.to_string(unit=u.hourangle, sep=' ', precision=2, pad=True) \
                         + coords.dec.to_string(sep=' ', precision=2, alwayssign=True, pad=True)
            try:
                objects = skybot.cone_search(coords, 30 * u.arcsec, time_obs.iso, location='645',
                                             get_raw_response=False, cache=False)
            except:
                print(f'{idx} It was error load error')
                # print(time_obs.iso, iso_coords)
                # temp['objID'] = df.iloc[idx]['objID'].astype(np.int64)
                if dobj_sky.empty:
                    dobj_sky['objID'] = df.iloc[idx:idx + 1]['objID'].astype(np.int64)
                else:
                    temp1 = pd.DataFrame({'objID': df.iloc[idx:idx + 1]['objID'].astype(np.int64)})
                    # print(temp1)
                    dobj_sky = dobj_sky.append(temp1, ignore_index=True)
            else:
                # print(dobj_sky)
                if dobj_sky.empty:
                    dobj_sky = objects.to_pandas()
                    dobj_sky['objID'] = df.iloc[idx]['objID'].astype(np.int64)
                else:
                    temp = objects.to_pandas()
                    print(idx, temp)
                    temp['objID'] = df.iloc[idx]['objID'].astype(np.int64)
                    dobj_sky = dobj_sky.append(temp, ignore_index=True)
            # print(dobj_sky)
        print(dobj_sky)
        dobj_sky.to_csv(f'{path}/init/last/sso_skybot_{b:04d}.csv')


def skybot_remove_duplicates():
    # remove duplicates
    df = pd.read_csv(f'{path}/init/last/sso_skybot_tot.csv')
    df0 = df[~df.duplicated("objID", keep=False)].sort_values("objID")
    print(df0)
    df1 = df[df.duplicated("objID", keep=False)].sort_values("objID")
    print(df1)
    df2 = df1.loc[df1.groupby("objID")["centerdist"].idxmin()]
    print(df2)
    df_out = pd.concat([df0, df2])
    del df_out['Unnamed: 0']
    # df_out['Number'].astype(int)
    df_out['Number'] = df_out['Number'].astype('Int64')
    print(df_out)

    df_out.to_csv(f'{path}/init/last/sso_skybot_tot1.csv', index=False)


def calc_r2():
    import sklearn.metrics
    import statsmodels.api as sm
    bands = ['r', 'i', 'u', 'z', 'g']
    w_init = np.array([1.0, 0.5, 0.25, 0.25, 1.0])
    # w_init = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    df = pd.read_csv(f'{path}/init/last/sso_tot3ez1.csv')
    r2 = np.zeros(df.__len__())
    ra = df[['offsetRa_r', 'offsetRa_i', 'offsetRa_u', 'offsetRa_z', 'offsetRa_g']].to_numpy()
    dec = df[['offsetDec_r', 'offsetDec_i', 'offsetDec_u', 'offsetDec_z', 'offsetDec_g']].to_numpy()
    # mag = df[['psfMagErr_r', 'psfMagErr_i', 'psfMagErr_u', 'psfMagErr_z', 'psfMagErr_g']].to_numpy()
    tm = df[['TAI_r', 'TAI_i', 'TAI_u', 'TAI_z', 'TAI_g']].to_numpy()

    def fit_poly_through_origin(x, y, n=1):
        a = x[:, np.newaxis] ** np.arange(1, n + 1)
        coeff = np.linalg.lstsq(a, y)[0]
        return np.concatenate(([0], coeff))

    def polyfit(x, y, degree):
        results = {}

        coeffs = np.ma.polyfit(x, y, degree)
        # c1 = fit_poly_through_origin(x, y, 1)
        print(coeffs)
        # print(c1)

        # Polynomial Coefficients
        results['polynomial'] = coeffs.tolist()
        # results['polynomial'] = c1.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # p = np.poly1d(c1)
        # fit values, and mean
        yhat = p(x)  # or [p(z) for z in x]
        ybar = np.sum(y) / len(y)  # or sum(y)/len(y)
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        results['determination'] = ssreg / sstot

        return results

    def coef_determ(coeffs, x, y):
        # r-squared
        p = np.poly1d(coeffs)
        delta = np.abs(y - p(x))
        mask_r = delta < (72 * coeffs[0])
        if mask_r.sum() > 2:
            # fit values, and mean
            xm = np.ma.array(data=x, mask=~mask_r)
            ym = np.ma.array(data=y, mask=~mask_r)
        else:
            xm = np.ma.array(data=x)
            ym = np.ma.array(data=y)

        # yhat = p(xm)  # or [p(z) for z in x]
        # ybar = np.ma.mean(ym)  # or sum(y)/len(y)
        # ssreg = np.ma.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        # sstot = np.ma.sum((ym - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        # r_value = ssreg / sstot

        pearson_r, pearson_p = stats.mstats.pearsonr(xm, ym)

        return pearson_r, xm, ym

    # for i in range(9932, 10000):
    for i in range(df.__len__()):
        # print(i)
        mask = np.array(df.iloc[i][['br', 'bi', 'bu', 'bz', 'bg']].to_numpy(), dtype=bool)
        xy = np.ma.array(data=np.sqrt(ra[i] ** 2 + dec[i] ** 2) * 60.0,
                         mask=~mask)
        tmi = np.ma.array(data=tm[i] - np.min(tm[i]), mask=~mask)

        if mask.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.mstats.linregress(tmi, xy)
            # print(slope, intercept, r_value, p_value, std_err)
            # r21, xm, ym = coef_determ([slope, intercept], tmi, xy)
            r21 = r_value ** 2
        else:
            r21 = 0.0

        mslope, mintercept = stats.mstats.siegelslopes(xy.data, tmi.data)
        r22, xm, ym = coef_determ([mslope, mintercept], tmi.data, xy.data)
        r22 = r22 ** 2
        if r22 < 0.5:
            r2[i] = np.max([r21, r22])
        else:
            r2[i] = r22

        if i % 10000 == 0:
            print(i)

        isshow = True
        if isshow:
            xx = np.linspace(tmi.data.min(), tmi.data.max(), 10)
            plt.plot(tmi.data, xy.data, 'ro', label='rejected Data \nby photometry')
            plt.plot(tmi, xy, 'bo', label='Data')
            plt.plot(xm, mintercept + mslope * xm, 'g.-', label=fr'$R^2=${r22:.2f}, Siegel')
            plt.plot(xm.data, mintercept + mslope * xm.data, 'g')
            plt.plot(xx, xx * slope + intercept, 'k-', label=fr'$R^2=${r21:.2f}, OLS')
            # plt.plot(tmi, filtered_data, 'ko', label='Fitted Model')
            # plt.plot(tmi, xy, "ko", label="Fitted Data")
            for j in range(tmi.shape[0]):
                plt.annotate(bands[j], xy=(tmi[j], xy[j]))
            plt.legend()
            plt.show()

    df['R2'] = r2
    # df.to_csv(f'{path}/init/last/sso_tot3f.csv', index=False)


def concat_by_key():
    df1 = pd.read_csv(f'{path}/init/last/sso_skybot_tot1.csv')
    df2 = pd.read_csv(f'{path}/init/last/sso_tot3b.csv')
    df3 = pd.merge(df1, df2, left_on='objID', right_on='objID', how='inner')
    df3['Number'] = df3['Number'].astype('Int64')
    print(df3)
    df3.to_csv(f'{path}/init/last/sso_tot3c.csv', index=False)


def remove_gaia():
    df1 = pd.read_csv(f'{path}/init/last/sso_tot3a.csv')
    print(df1)
    df2 = pd.read_csv(f'{path}/init/gaia_cross.csv')
    print(df2)
    a = df2['objID'].to_list()
    print(a)
    df = df1[~df1['objID'].isin(a)]
    print(df)

    df.to_csv(f'{path}/init/last/sso_tot3b.csv', index=False)


def skybot_analisis():
    df = pd.read_csv(f'{path}/init/last/sso_tot3f.csv')
    dfsky = df[df['V'] > 0]
    types = list(set(dfsky['Type']))
    print(types)
    print(f'Comets:{len(dfsky[dfsky["Type"].str.contains("Comet")])}')
    print(f'KBO:{len(dfsky[dfsky["Type"].str.contains("KBO")])}')
    print(f'NEA:{len(dfsky[dfsky["Type"].str.contains("NEA")])}')
    print(f'Main:{len(dfsky[dfsky["Type"].str.contains("MB")])}')
    print(f'Mars-Crosser:{len(dfsky[dfsky["Type"].str.contains("Mars-Crosser")])}')
    print(f'Trojan:{len(dfsky[dfsky["Type"].str.contains("Trojan")])}')
    print(f'Hungaria:{len(dfsky[dfsky["Type"].str.contains("Hungaria")])}')

    # print(f'Comets:{len(dfsky["Type"] == "Comet")}')
    # print(dfsky)
    bcenter = False
    if bcenter:
        print(dfsky['centerdist'].median())
        print(len(dfsky['Name'].unique()))
        dp = dfsky[dfsky['centerdist'] < 3]['centerdist']
        ax = dp.plot.hist(bins=50, xlim=(0, 3), alpha=0.8, rwidth=1.0, color=u'#ff7f0e',
                          # title='Distance distribution between SkyBot and SDSS coordinates',
                          label='center distance')
        ax.set_xlabel('Distance, (arcsec)')
        ax.legend()
        plt.savefig('./figs/distance.png', dpi=300)
        plt.show()

    if bcenter:
        V_SDSS = pd.DataFrame(df['psfMag_g'] - 0.69 * (df['psfMag_g'] - df['psfMag_r']) - 0.01, columns=['SDSS'])
        print(V_SDSS)
        V_SDSS['SkyBot'] = dfsky['V']
        # d_mag = pd.merge([V_sky, V_SDSS])
        print(V_SDSS)
        ax = V_SDSS.plot.hist(bins=100, alpha=0.5, xlim=(15, 24))
        ax.set_xlabel('V, (mag)')
        plt.savefig('./figs/skybot_magV.png', dpi=300)
        plt.show()

    if not bcenter:
        r21 = df['R2'].to_numpy()
        # r21 = r21[(r21 > 0.8) & (r21 < 1.4)]
        # bins = np.arange(np.floor(r21.min()), np.ceil(r21.max()))
        values1, base1 = np.histogram(r21, bins=100, range=(0.8, 1.1))
        cumulative = np.cumsum(values1)
        plt.plot(base1[:-1], cumulative / cumulative.max(), label=r'$R^2$ Total', lw=2)
        r22 = dfsky['R2'].to_numpy()
        # r22 = r22[(r22 > 0.8) & (r22 < 1.4)]
        values2, base2 = np.histogram(r22, bins=100, range=(0.8, 1.1))
        cumulative = np.cumsum(values2)
        plt.plot(base2[:-1], cumulative / cumulative.max(), label=r'$R^2$ Known', lw=2)
        plt.legend(loc=2)
        plt.grid(True)
        print(df[df['R2'] > 0.90].__len__() / df.__len__())

        # plt.savefig('./figs/R2.png', dpi=300)
        plt.show()


def adr4_tot3f_compare():
    df_tot = pd.read_csv(f'{path}/init/last/sso_tot3f.csv')
    dis_tot = df_tot['TAI_r'].to_numpy()
    print('Total loaded.')
    df_adr4 = pd.read_csv(f'{path}/init/ADR4.csv')
    print('ADR4 loaded.')
    dis_adr4 = df_adr4['col8'].to_numpy() * 3600 * 24.0
    c_adr4 = SkyCoord(ra=df_adr4['col9'] * u.deg,
                      dec=df_adr4['col10'] * u.deg,
                      distance=dis_adr4 * u.m)

    max_sep = 1.0 * u.arcsec
    max_sep2 = 3600.0 * u.m
    lb = 1100000
    step = 100000
    df_tot = df_tot[lb:lb + step]
    l = df_tot.__len__()
    print(l)
    b_adr4 = np.zeros(shape=l, dtype=bool)
    for i in range(0, l):
        # for i in range(10):
        c = SkyCoord(ra=df_tot.iloc[i]['ra'] * u.deg,
                     dec=df_tot.iloc[i]['dec'] * u.deg,
                     distance=dis_tot[i] * u.m)
        d2_sky = c.separation(c_adr4)
        d2_3d = c.separation_3d(c_adr4)
        cond = np.any(d2_sky < max_sep) & np.any(d2_3d < max_sep2)
        b_adr4[i] = cond
        if (i % 100 == 0):
            print(i + lb, b_adr4[b_adr4].shape)

    # df_tot['adr4'] = b_adr4
    # print(df_tot.iloc[:20]['adr4'])
    np.save('./data/b_adr4_5a3.npy', b_adr4)
    # df_tot.to_csv(f'{path}/init/last/sso_tot3d.csv', index=False)

    cols = df_tot.columns.tolist()
    print(cols)
    cols = cols[-1:] + cols[:-1]
    print(cols)
    df_tot = df_tot[cols]

    #
    # d2d = c_adr4.separation(c_adr4)
    # catalogmsk = (d2d < 1 * u.arcsec) # & (d2d > 0.001 * u.arcsec)
    # print(catalogmsk.shape)
    # idxcatalog = np.where(catalogmsk)[0]
    # print(idxcatalog)
    # # print(idxcatalog.shape)
    #
    # #
    # # print(c_adr4[:10])
    # # dis_tot = df_tot['TAI_r'].to_numpy()
    # # c_tot = SkyCoord(ra=df_tot['ra'] * u.deg, dec=df_tot['dec'] * u.deg,
    # #                  distance=dis_tot * u.m)
    # # # print(c_tot[:10])
    # # #
    # # # sep = 1.0 * u.arcsec
    # # # idx, d2d, d3d = c_adr4.match_to_catalog_3d(c_tot)
    # # # sep_constraint = d2d < sep
    # # # c_matches = c[sep_constraint]
    # # # print(idx[:10], sep[:10])
    # #
    # # # idx_adr4, d2d, d3d = c_adr4.match_to_catalog_3d(c_tot)
    # # # max_sep = 1.0 * u.arcsec
    # # # sep_constraint = d2d < max_sep
    # # # print(sep_constraint.shape)
    # # # c_adr4_match = c_tot[sep_constraint]
    # # # print(c_adr4_match[:100])
    # #
    # # idx_adr4, idx_tot, d2d, d3d = c_tot.search_around_sky(c_adr4, 1 * u.m)
    # # print(idx_adr4.shape)
    # # print(idx_tot[:10])
    #
    # # d2d = c_tot.separation(c_adr4)
    # # c_tot_mask = d2d < 1 * u.arcsec
    # # idxcatalog = np.where(c_tot_mask)
    # # print(idxcatalog[:10])


def concat_numpy():
    df_tot = pd.read_csv(f'{path}/init/last/sso_tot3f.csv')
    a = []
    for i in range(0, 5):
        b = np.load(f'./data/b_adr4_{i + 1}.npy')
        a.append(b)

    b = np.load(f'./data/b_adr4_{5}.npy')
    a.append(b)

    b_adr4 = np.concatenate((a[0], a[1], a[2], a[3], a[4]))
    print(b_adr4.shape)
    df_tot['adr4'] = b_adr4
    cols = df_tot.columns.tolist()
    print(cols)
    cols = cols[:1] + cols[22:-11] + cols[1:22] + cols[-11:]
    print(cols)
    df_tot = df_tot[cols]
    df_tot = format_adr(df_tot)

    df_tot.to_csv(f'{path}/init/last/sso_tot3g1.csv', index=False)


def cat_correction1():
    df = pd.read_csv(f'{path}/init/last/sso_tot3g.csv')
    df = df.rename({'RA': 'ra_sb', 'DEC': 'dec_sb', 'bo': 'boffset',
                    'bu': 'bphot_u', 'bg': 'bphot_g', 'br': 'bphot_r', 'bi': 'bphot_i', 'bz': 'bphot_z',
                    'bru': 'bcolor_ru', 'brg': 'bcolor_rg', 'bri': 'bcolor_ri', 'brz': 'bcolor_rz'},
                   axis='columns')
    df.to_csv(f'{path}/init/last/sso_tot4.csv', index=False)

    print(df.head())


def linarity():
    # df = pd.read_csv(f'{path}/init/last/sso_tot4.csv', nrows=100000)
    df = pd.read_csv(f'{path}/init/last/sso_tot4.csv')
    # cond = (df['offsetRa_g'] < 3) & (df['offsetRa_i'] < 3) & (df['offsetRa_z'] < 3) & (df['offsetRa_u'] < 3) &\
    #        (df['offsetRa_g'] > -3) & (df['offsetRa_i'] > -3) & (df['offsetRa_z'] > -3) & (df['offsetRa_u'] > -3)
    # df = df[cond]
    # x, y = df['offsetRa_g'], df[['offsetRa_i', 'offsetRa_z']]
    model = np.polyfit(df['offsetRa_g'].to_numpy(), df['offsetRa_i'].to_numpy(), 1)
    # print(model)
    gi = df['offsetRa_i'] - df['offsetRa_g'] * 0.25
    sgi = np.std(gi)
    gz = df['offsetRa_z'] - df['offsetRa_g'] * 0.75
    sgz = np.std(gz)
    l = 1
    # gu = (df['offsetRa_u'] - df['offsetRa_g'] * 0.5).to_numpy()
    gu = df['offsetRa_u'] - df['offsetRa_g'] * 0.5
    sgu = np.std(gu)

    i = 0
    while (l > 0) and i < 100:
        cond_u = (gu < 3 * sgu) & (gu > -3 * sgu)
        cond_i = (gi < 3 * sgi) & (gi > -3 * sgi)
        cond_z = (gz < 3 * sgz) & (gz > -3 * sgz)
        l = gu[~cond_u].shape[0] + gi[~cond_i].shape[0] + gz[~cond_z].shape[0]
        gu = gu[cond_u]
        gi = gi[cond_i]
        gz = gz[cond_z]
        sgu = np.std(gu)
        sgi = np.std(gi)
        sgz = np.std(gz)

        i += 1
        print(i, sgu, sgi, sgz, l)
    print(gi.__len__(), gz.__len__(), gu.index.__len__())
    print((gi.index & gz.index & gu.index))

    # plt.scatter(df['offsetRa_g'][gi.index][::5], gi[gi.index][::5], marker='.', alpha=0.5)
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [3*sgi, 3*sgi], c='k')
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [-3*sgi, -3*sgi], c='k')
    # plt.scatter(df['offsetRa_g'][gz.index][::5], gz[gz.index][::5], marker='.', alpha=0.5)
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [3*sgz, 3*sgz], c='k')
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [-3*sgz, -3*sgz], c='k')
    # plt.scatter(df['offsetRa_g'][gu.index][::5], gu[gu.index][::5], marker='.', alpha=0.5)
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [3*sgu, 3*sgu], c='k')
    # plt.plot([df['offsetRa_g'].min(), df['offsetRa_g'].max()], [-3*sgu, -3*sgu], c='k')
    # plt.show()

    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    #
    # prestige_model = ols("offsetRa_g ~ offsetRa_i + offsetRa_z + offsetRa_u", data=df).fit()
    # print(prestige_model.summary())
    # fig = plt.figure(figsize=(12, 8))
    # sm.graphics.plot_partregress_grid(prestige_model, fig=fig)
    # plt.show()
    #
    # y, X = df['offsetRa_g'][], df[['offsetRa_i', 'offsetRa_z']]
    # y1, X1 = df['offsetRa_g'], df['offsetRa_i']
    # y2, X2 = df['offsetRa_g'], df['offsetRa_z']
    # #                              # 'offsetDec_i', 'offsetDec_u', 'offsetDec_z']]
    # xx = np.linspace(-3, 3, 2)
    # # # print(xx)
    # # res = sm.OLS(y, X).fit()
    # # print(res.summary())
    # res1 = sm.OLS(y1, X1).fit()
    # print(res1.summary())
    # res2 = sm.OLS(y2, X2).fit()
    # print(res2.summary())
    # # # for i in range(1):
    # # #     outliers = res.outlier_test(method='bonf', alpha=0.1)
    # # #     cond = outliers['bonf(p)'] < 0.9
    # # #     # print(outliers[cond])
    # # #     y = y.drop(outliers[cond].index[:])
    # # #     # print(y)
    # # #     X = X.drop(outliers[cond].index[:])
    # # #     res = sm.OLS(y, X).fit()
    # # #     print(res.summary2())
    # # #
    # # #     ax.scatter(df.iloc[outliers[cond].index]['offsetRa_g'],
    # # #                df.iloc[outliers[cond].index]['offsetRa_i'],
    # # #                df.iloc[outliers[cond].index]['offsetRa_z'], marker='o', color='black')
    # # #
    # #     # print(y, X)
    # #
    # #     # print(res.params)
    # dy = res1.params['offsetRa_i']
    # dz = res2.params['offsetRa_z']
    # # # print(dy, dz)
    #
    # fig = plt.figure(figsize=(10, 10))
    # # # ax = Axes3D(fig)
    # ax = fig.add_subplot(111)
    # # # ax.scatter(df['offsetRa_g'], df['offsetRa_i'], df['offsetRa_z'], marker='.', c=df['offsetRa_u'], cmap='Blues')
    # ax.scatter(df['offsetRa_g'], df['offsetRa_i'], marker='.', c=df['offsetRa_u'], cmap='Blues')
    # ax.scatter(df['offsetRa_g'], df['offsetRa_z'], marker='.', c=df['offsetRa_u'], cmap='Oranges')
    # # # ax.scatter(df['offsetDec_g'], df['offsetDec_i'], df['offsetDec_z'], marker='.', c=df['offsetDec_u'], cmap='Oranges')
    # # ax.set_xlabel('g')
    # # ax.set_ylabel('i')
    # # # ax.set_zlabel('z')
    # # ax.set_xlim(-3, 3)
    # # ax.set_ylim(-3, 3)
    # ax.plot([xx[0], xx[-1]],
    #         [xx[0]/dy, xx[-1]/dy],
    #         # [xx[0]/dz, xx[-1]/dz],
    #         color='black', lw=3)
    # ax.plot([xx[0], xx[-1]],
    #         # [xx[0] / dy, xx[-1] / dy],
    #         [xx[0]/dz, xx[-1]/dz],
    #         color='black', lw=3)
    # plt.show()


def matrix_offset():
    k = 200
    df = pd.read_csv(f'{path}/init/last/sso_tot4.csv', nrows=k)
    a = np.zeros((5, 5, k))
    b = np.zeros((5, 5, k))
    bands = {'r': 0, 'i': 1, 'u': 2, 'z': 3, 'g': 4}
    bands_col = ['i', 'u', 'z', 'g']

    # a[0, 0] = df['offsetRa_r'] - df['offsetRa_r'] * (bands['r'] / bands['r'])
    # a[0, 1] = df['offsetRa_i'] - df['offsetRa_r'] * (bands['i'] / bands['r'])
    # a[0, 2] = df['offsetRa_u'] - df['offsetRa_r'] * (bands['u'] / bands['r'])
    # a[0, 3] = df['offsetRa_z'] - df['offsetRa_r'] * (bands['z'] / bands['r'])
    # a[0, 4] = df['offsetRa_g'] - df['offsetRa_r'] * (bands['g'] / bands['r'])

    a[1, 0] = df['offsetRa_r'] - df['offsetRa_i'] * (bands['r'] / bands['i'])
    a[1, 1] = df['offsetRa_i'] - df['offsetRa_i'] * (bands['i'] / bands['i'])
    a[1, 2] = df['offsetRa_u'] - df['offsetRa_i'] * (bands['u'] / bands['i'])
    a[1, 3] = df['offsetRa_z'] - df['offsetRa_i'] * (bands['z'] / bands['i'])
    a[1, 4] = df['offsetRa_g'] - df['offsetRa_i'] * (bands['g'] / bands['i'])

    a[2, 0] = df['offsetRa_r'] - df['offsetRa_u'] * (bands['r'] / bands['u'])
    a[2, 1] = df['offsetRa_i'] - df['offsetRa_u'] * (bands['i'] / bands['u'])
    a[2, 2] = df['offsetRa_u'] - df['offsetRa_u'] * (bands['u'] / bands['u'])
    a[2, 3] = df['offsetRa_z'] - df['offsetRa_u'] * (bands['z'] / bands['u'])
    a[2, 4] = df['offsetRa_g'] - df['offsetRa_u'] * (bands['g'] / bands['u'])

    a[3, 0] = df['offsetRa_r'] - df['offsetRa_z'] * (bands['r'] / bands['z'])
    a[3, 1] = df['offsetRa_i'] - df['offsetRa_z'] * (bands['i'] / bands['z'])
    a[3, 2] = df['offsetRa_u'] - df['offsetRa_z'] * (bands['u'] / bands['z'])
    a[3, 3] = df['offsetRa_z'] - df['offsetRa_z'] * (bands['z'] / bands['z'])
    a[3, 4] = df['offsetRa_g'] - df['offsetRa_z'] * (bands['g'] / bands['z'])

    a[4, 0] = df['offsetRa_r'] - df['offsetRa_g'] * (bands['r'] / bands['g'])
    a[4, 1] = df['offsetRa_i'] - df['offsetRa_g'] * (bands['i'] / bands['g'])
    a[4, 2] = df['offsetRa_u'] - df['offsetRa_g'] * (bands['u'] / bands['g'])
    a[4, 3] = df['offsetRa_z'] - df['offsetRa_g'] * (bands['z'] / bands['g'])
    a[4, 4] = df['offsetRa_g'] - df['offsetRa_g'] * (bands['g'] / bands['g'])

    # a[0, 0] = df['offsetDec_r'] - df['offsetDec_r'] * (bands['r'] / bands['r'])
    # a[0, 1] = df['offsetDec_i'] - df['offsetDec_r'] * (bands['i'] / bands['r'])
    # a[0, 2] = df['offsetDec_u'] - df['offsetDec_r'] * (bands['u'] / bands['r'])
    # a[0, 3] = df['offsetDec_z'] - df['offsetDec_r'] * (bands['z'] / bands['r'])
    # a[0, 4] = df['offsetDec_g'] - df['offsetDec_r'] * (bands['g'] / bands['r'])

    b[1, 0] = df['offsetDec_r'] - df['offsetDec_i'] * (bands['r'] / bands['i'])
    b[1, 1] = df['offsetDec_i'] - df['offsetDec_i'] * (bands['i'] / bands['i'])
    b[1, 2] = df['offsetDec_u'] - df['offsetDec_i'] * (bands['u'] / bands['i'])
    b[1, 3] = df['offsetDec_z'] - df['offsetDec_i'] * (bands['z'] / bands['i'])
    b[1, 4] = df['offsetDec_g'] - df['offsetDec_i'] * (bands['g'] / bands['i'])

    b[2, 0] = df['offsetDec_r'] - df['offsetDec_u'] * (bands['r'] / bands['u'])
    b[2, 1] = df['offsetDec_i'] - df['offsetDec_u'] * (bands['i'] / bands['u'])
    b[2, 2] = df['offsetDec_u'] - df['offsetDec_u'] * (bands['u'] / bands['u'])
    b[2, 3] = df['offsetDec_z'] - df['offsetDec_u'] * (bands['z'] / bands['u'])
    b[2, 4] = df['offsetDec_g'] - df['offsetDec_u'] * (bands['g'] / bands['u'])

    b[3, 0] = df['offsetDec_r'] - df['offsetDec_z'] * (bands['r'] / bands['z'])
    b[3, 1] = df['offsetDec_i'] - df['offsetDec_z'] * (bands['i'] / bands['z'])
    b[3, 2] = df['offsetDec_u'] - df['offsetDec_z'] * (bands['u'] / bands['z'])
    b[3, 3] = df['offsetDec_z'] - df['offsetDec_z'] * (bands['z'] / bands['z'])
    b[3, 4] = df['offsetDec_g'] - df['offsetDec_z'] * (bands['g'] / bands['z'])

    b[4, 0] = df['offsetDec_r'] - df['offsetDec_g'] * (bands['r'] / bands['g'])
    b[4, 1] = df['offsetDec_i'] - df['offsetDec_g'] * (bands['i'] / bands['g'])
    b[4, 2] = df['offsetDec_u'] - df['offsetDec_g'] * (bands['u'] / bands['g'])
    b[4, 3] = df['offsetDec_z'] - df['offsetDec_g'] * (bands['z'] / bands['g'])
    b[4, 4] = df['offsetDec_g'] - df['offsetDec_g'] * (bands['g'] / bands['g'])

    # print(a[:, :, idx])
    xx = np.zeros(5)
    yy = np.zeros(5)

    for idx in range(100, k):
        print(idx)
        x = a[1:, 1:, idx]
        y = b[1:, 1:, idx]
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print(x)
        print(y)
        # print(np.median(x) + x.sum(axis=1))

        gx = np.concatenate((np.abs(x[:, 3]), np.abs(x[3, :]), np.abs(y[:, 3]), np.abs(y[3, :])), axis=0)
        zx = np.concatenate((np.abs(x[:, 2]), np.abs(x[2, :]), np.abs(y[:, 2]), np.abs(y[2, :])), axis=0)
        ux = np.concatenate((np.abs(x[:, 1]), np.abs(x[1, :]), np.abs(y[:, 1]), np.abs(y[1, :])), axis=0)
        ix = np.concatenate((np.abs(x[:, 0]), np.abs(x[0, :]), np.abs(y[:, 0]), np.abs(y[0, :])), axis=0)

        # gy = np.concatenate((np.abs(y[:, 3]), np.abs(y[3, :])), axis=0)
        # zy = np.concatenate((np.abs(y[:, 2]), np.abs(y[2, :])), axis=0)
        # uy = np.concatenate((np.abs(y[:, 1]), np.abs(y[1, :])), axis=0)
        # iy = np.concatenate((np.abs(y[:, 0]), np.abs(y[0, :])), axis=0)

        print(gx, zx, ix, ux)
        print(np.median(gx), np.median(zx), np.median(ix), np.median(ux))

        # print(x[:, 1], x[1, :])
        # print(x[:, 0], x[0, :])
        #
        smed = []
        sa = []
        sb = []

        xs, ys = 0.0, 0.0
        for i in range(0, 4):
            for j in range(0, 4):
                if i != j:
                    # sa.append((x[i, j]**2, x[j, i]**2))
                    sa.append((np.absolute(x[i, j]), np.absolute(x[j, i])))
                    sb.append((np.absolute(y[i, j]), np.absolute(y[j, i])))
                    # xs + = np.abs(x[i, j]) +

        # print(sa)
        smed.append((np.median(sa), np.median(sb)))
        # smed.append(np.median(sa))

        print(np.median(sa))
        print(np.median(sb))
        # smed.append(np.median(sa)+np.median(sb))

        # print(np.median(sa)+np.median(sb), np.median(sa), np.median(sb))
        # r_val = np.median(smed)
        # smed = list(smed - r_val)
        # smed.insert(0, r_val)
        print(bands_col)
        print(smed)

        for i, key_i in enumerate(bands.keys()):
            # print(np.median(x[i]))
            # key = f'offsetRa_{key_i}'
            # plt.scatter(key_i, df.iloc[idx][f'offsetRa_{key_i}'])
            # plt.scatter(key_i, df.iloc[idx][f'offsetDec_{key_i}'])
            plt.scatter(df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}'])
            plt.annotate(key_i,
                         xy=(df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}']))
        #     xx[i] = df.iloc[idx][f'offsetRa_{key_i}']
        #     yy[i] = df.iloc[idx][f'offsetDec_{key_i}']
        #
        # mslope, mintercept, _, _, _ = stats.linregress(xx, yy, weight)
        # plt.plot(xx, xx * mslope + mintercept, 'k-')
        plt.xlim(-4, 4)
        plt.ylim(-3, 3)
        plt.show()

    # b = np.zeros((5, 5))
    # for i, key_i in enumerate(bands.keys()):
    #     for j, key_j in enumerate(bands.keys()):
    #         if j > 0:
    #             b[i, j] = i / j
    # print(i, key)
    # for key_j in bands:
    #     print(key_i, key_j)
    #     # b[i, j] = key_i / bands.keys()[j]

    # print(b)


def linarity2():
    # k = 200
    rad2 = 0.4 ** 2
    # df = pd.read_csv(f'{path}init/last/sso_tot4.csv', nrows=k)
    df = pd.read_csv(f'{path}init/last/sso_tot4.csv')
    isshow = True
    # bi = 1
    # step = 100000
    beg = 0
    # end = (bi + 1) * step
    cz = np.zeros(df.__len__())
    cu = np.zeros(df.__len__())
    ci = np.zeros(df.__len__())

    for idx in range(df.__len__()):
        idc = idx - beg
        if idc % 1000 == True:
            print(idc, idx)
        g_ra = df.iloc[idx][f'offsetRa_g']
        g_dec = df.iloc[idx][f'offsetDec_g']
        h_ra = g_ra / 4
        h_dec = g_dec / 4

        z_ra, z_dec = h_ra * 3, h_dec * 3
        u_ra, u_dec = h_ra * 2, h_dec * 2
        i_ra, i_dec = h_ra * 1, h_dec * 1

        dz2 = (z_ra - df.iloc[idx][f'offsetRa_z']) ** 2 + (z_dec - df.iloc[idx][f'offsetDec_z']) ** 2
        du2 = (u_ra - df.iloc[idx][f'offsetRa_u']) ** 2 + (u_dec - df.iloc[idx][f'offsetDec_u']) ** 2
        di2 = (i_ra - df.iloc[idx][f'offsetRa_i']) ** 2 + (i_dec - df.iloc[idx][f'offsetDec_i']) ** 2

        cz[idc] = dz2 < rad2
        cu[idc] = du2 < rad2
        ci[idc] = di2 < rad2
        # print(cz[idc], cu[idc], ci[idc])

        if isshow:
            bands = {'r': 0, 'i': 1, 'u': 2, 'z': 3, 'g': 4}
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            for i, key_i in enumerate(bands.keys()):
                plt.scatter(df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}'])
                plt.annotate(key_i,
                             xy=(df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}']))
            if cz[idx - beg]:
                col = 'Grey'
            else:
                col = 'Red'
            circle_z = plt.Circle((z_ra, z_dec), np.sqrt(rad2), color=col, fill=False)
            if cu[idx - beg]:
                col = 'Grey'
            else:
                col = 'Red'
            circle_u = plt.Circle((u_ra, u_dec), np.sqrt(rad2), color=col, fill=False)
            if ci[idx - beg]:
                col = 'Grey'
            else:
                col = 'Red'
            circle_i = plt.Circle((i_ra, i_dec), np.sqrt(rad2), color=col, fill=False)

            plt.plot([0, i_ra, u_ra, z_ra, g_ra], [0, i_dec, u_dec, z_dec, g_dec], 'b.-')
            ax.add_artist(circle_z)
            ax.add_artist(circle_u)
            ax.add_artist(circle_i)
            plt.plot([i_ra, df.iloc[idx][f'offsetRa_i']], [i_dec, df.iloc[idx][f'offsetDec_i']],
                     color='Black')
            plt.plot([u_ra, df.iloc[idx][f'offsetRa_u']], [u_dec, df.iloc[idx][f'offsetDec_u']],
                     color='Black')
            plt.plot([z_ra, df.iloc[idx][f'offsetRa_z']], [z_dec, df.iloc[idx][f'offsetDec_z']],
                     color='Black')

            # plt.scatter([0, i_ra, u_ra, z_ra, g_ra], [0, i_dec, u_dec, z_dec, g_dec], s=100,
            #             edgecolors='Grey', facecolors='none')
            plt.gca().set_aspect('equal')
            plt.xlim(df.iloc[idx][f'offsetRa_g'] + np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetRa_g']),
                     df.iloc[idx][f'offsetRa_r'] - np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetRa_g']))
            plt.ylim(df.iloc[idx][f'offsetDec_g'] + np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetDec_g']),
                     df.iloc[idx][f'offsetDec_r'] - np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetDec_g']))
            # plt.ylim(-2, 2)
            plt.show()
    # c = np.concatenate((cz, cu, ci))
    c = np.stack((cz, cu, ci), axis=-1)
    # print(c)
    # np.save(f'./data/c_astrometry.npy', c)


def taxonomy_show():
    from matplotlib.patches import Ellipse
    import matplotlib.cm as cm

    df = pd.read_csv('./data/class_colors.csv')
    print(df)
    plt.figure()
    ax = plt.gca()
    colors = cm.rainbow(np.linspace(0, 1, df.__len__()))
    for i in range(df.__len__()):
        x = df.iloc[i]['g-u']
        y = df.iloc[i]['i-z']
        sx = df.iloc[i]['sigma g-u']
        sy = df.iloc[i]['sigma i-z']
        print(x, y, sx, sy)
        ellipse = Ellipse(xy=(x, y),
                          width=sx,
                          height=sy,
                          edgecolor=colors[i],
                          facecolor='None',
                          lw=1, label=df.iloc[i]['class'])
        ax.add_patch(ellipse)
        # circle = plt.((df.iloc[i]['g-u'], df.iloc[i]['i-z']), )
    plt.scatter(df['g-u'], df['i-z'])
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


def astrometry_add():
    df = pd.read_csv(f'{path}init/last/sso_tot4.csv')
    df = df.rename(columns={'RA_rate': 'ra_sb_rate'})
    df = df.rename(columns={'DEC_rate': 'dec_sb_rate'})

    c = np.load(f'./data/c_astrometry.npy')
    # print(c[:, 0])
    # print(c)
    df['bastrom_r'] = np.ones(df.__len__(), dtype=bool)
    df['bastrom_i'] = c[:, 2].astype(bool)
    df['bastrom_u'] = c[:, 1].astype(bool)
    df['bastrom_z'] = c[:, 0].astype(bool)
    df['bastrom_g'] = np.ones(df.__len__(), dtype=bool)
    print(df)

    df = format_cat(df)
    df.to_csv(f'{path}/init/last/sso_tot4a.csv', index=False)


def format_cat(adr5):
    adr5['psfMagErr_u'] = adr5['psfMagErr_u'].round(5)
    adr5['psfMagErr_g'] = adr5['psfMagErr_g'].round(5)
    adr5['psfMagErr_r'] = adr5['psfMagErr_r'].round(5)
    adr5['psfMagErr_i'] = adr5['psfMagErr_i'].round(5)
    adr5['psfMagErr_z'] = adr5['psfMagErr_z'].round(5)
    adr5['psfMag_u'] = adr5['psfMag_u'].round(5)
    adr5['psfMag_g'] = adr5['psfMag_g'].round(5)
    adr5['psfMag_r'] = adr5['psfMag_r'].round(5)
    adr5['psfMag_i'] = adr5['psfMag_i'].round(5)
    adr5['psfMag_z'] = adr5['psfMag_z'].round(5)
    adr5['rowv'] = adr5['rowv'].round(5)
    adr5['rowvErr'] = adr5['rowvErr'].round(5)
    adr5['colv'] = adr5['colv'].round(5)
    adr5['colvErr'] = adr5['colvErr'].round(5)
    adr5['vel'] = adr5['vel'].round(5)
    adr5['velErr'] = adr5['velErr'].round(5)
    # adr5['ang_dist'] = adr5['ang_dist'].round(3)
    # adr5['dv_dec'] = adr5['dv_dec'].round(3)
    # adr5['dv_ra'] = adr5['dv_ra'].round(3)
    # adr5['dv_abs'] = adr5['dv_abs'].round(6)
    # adr5['mjd'] = adr5['mjd'].round(6)

    adr5['offsetRa_u'] = adr5['offsetRa_u'].round(5)
    adr5['offsetRa_g'] = adr5['offsetRa_g'].round(5)
    adr5['offsetRa_r'] = adr5['offsetRa_r'].round(5)
    adr5['offsetRa_i'] = adr5['offsetRa_i'].round(5)
    adr5['offsetRa_z'] = adr5['offsetRa_z'].round(5)
    adr5['offsetDec_u'] = adr5['offsetDec_u'].round(5)
    adr5['offsetDec_g'] = adr5['offsetDec_g'].round(5)
    adr5['offsetDec_r'] = adr5['offsetDec_r'].round(5)
    adr5['offsetDec_i'] = adr5['offsetDec_i'].round(5)
    adr5['offsetDec_z'] = adr5['offsetDec_z'].round(5)
    adr5['R2'] = adr5['R2'].round(2)

    adr5['ra_sb'] = adr5['ra_sb'].round(8)
    adr5['dec_sb'] = adr5['dec_sb'].round(8)
    adr5['ra'] = adr5['ra'].round(8)
    adr5['dec'] = adr5['dec'].round(8)

    adr5['ra_sb_rate'] = adr5['ra_sb_rate'].round(8)
    adr5['dec_sb_rate'] = adr5['dec_sb_rate'].round(8)
    adr5['x'] = adr5['x'].round(8)
    adr5['y'] = adr5['y'].round(8)
    adr5['z'] = adr5['z'].round(8)
    adr5['vx'] = adr5['vx'].round(8)
    adr5['vy'] = adr5['vy'].round(8)
    adr5['vz'] = adr5['vz'].round(8)

    adr5['posunc'] = adr5['posunc'].round(3)
    adr5['centerdist'] = adr5['centerdist'].round(3)
    adr5['geodist'] = adr5['geodist'].round(8)
    adr5['heliodist'] = adr5['heliodist'].round(8)

    adr5['Number'] = adr5['Number'].astype('Int64')
    adr5['type'] = adr5['type'].astype(np.int)

    return adr5


def cat_correction2():
    df = pd.read_csv(f'{path}/init/last/sso_tot4a.csv')
    df = df.rename(columns={'Type': 'class'})
    df = df.rename(columns={'type_r': 'type'})
    df = df.rename(columns={'adr4': 'badr4'})

    a = np.zeros(df.__len__(), dtype=float)
    df.insert(24, 'rowvErr', a)
    df.insert(26, 'colvErr', a)
    df.insert(28, 'velErr', a)

    print(df[['objID', 'rowv', 'rowvErr', 'colv', 'colvErr', 'vel', 'velErr']])

    df16 = pd.read_csv(f'{path}/init/last/move16.csv')
    for i in range(df16.__len__()):
        if i % 1000 == 0:
            print(i)
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'rowv'] = df16.iloc[i]['rowv']
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'rowvErr'] = df16.iloc[i]['rowvErr']
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'colv'] = df16.iloc[i]['colv']
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'colvErr'] = df16.iloc[i]['colvErr']
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'vel'] = \
            np.sqrt(df16.iloc[i]['colv'] ** 2 + df16.iloc[i]['rowv'] ** 2)
        df.loc[df['objID'] == df16.iloc[i]['objID'], 'velErr'] = \
            np.sqrt(df16.iloc[i]['colvErr'] ** 2 + df16.iloc[i]['rowvErr'] ** 2)

    print(df[['objID', 'rowv', 'rowvErr', 'colv', 'colvErr', 'vel', 'velErr']])

    df82 = pd.read_csv(f'{path}/init/last/move82.csv')
    for i in range(df82.__len__()):
        if i % 1000 == 0:
            print(i)
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'rowv'] = df82.iloc[i]['rowv']
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'rowvErr'] = df82.iloc[i]['rowvErr']
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'colv'] = df82.iloc[i]['colv']
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'colvErr'] = df82.iloc[i]['colvErr']
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'vel'] = \
            np.sqrt(df82.iloc[i]['colv'] ** 2 + df82.iloc[i]['rowv'] ** 2)
        df.loc[df['objID'] == df82.iloc[i]['objID'], 'velErr'] = \
            np.sqrt(df82.iloc[i]['colvErr'] ** 2 + df82.iloc[i]['rowvErr'] ** 2)

    print(df[['objID', 'rowv', 'rowvErr', 'colv', 'colvErr', 'vel', 'velErr']])

    # cond = df['ra_sb'].isna()
    # print(df['ra_sb'])
    # print(cond)
    # df['bknown'] = ~cond
    # print(df)
    # df.to_csv(f'{path}/init/last/sso_tot4c.csv', index=False)

    # df = format_cat(df)
    # df.to_csv(f'{path}/init/last/sso_tot4b.csv', index=False)


def cat_correction3():
    df = pd.read_csv(f'{path}/init/last/sso_tot4b.csv')
    df = df.rename(columns={'bcolor_rg': 'bcolor_ru'})
    df = df.rename(columns={'bcolor_rg.1': 'bcolor_rg'})
    cond = df['ra_sb'].isna()
    # print(df['ra_sb'])

    df['bknown'] = ~cond
    cond_u = (np.abs(df['offsetRa_u']) > 1e3) | (np.abs(df['offsetDec_u']) > 1e3)
    print(df[df['bastrom_u'] == True].__len__())
    print(df['bastrom_u'][cond_u])
    df['bastrom_u'] = df['bastrom_u'] & ~cond_u
    print(df['bastrom_u'][cond_u])
    print(df[df['bastrom_u'] == True].__len__(), end='\n\n')

    print(df[df['bastrom_g'] == True].__len__())
    cond_g = (np.abs(df['offsetRa_g']) > 1e3) | (np.abs(df['offsetDec_g']) > 1e3)
    print(df['bastrom_g'][cond_g])
    df['bastrom_g'] = df['bastrom_g'] & ~cond_g
    print(df['bastrom_g'][cond_g])
    print(df[df['bastrom_g'] == True].__len__(), end='\n\n')

    print(df[df['bastrom_r'] == True].__len__())
    cond_r = (np.abs(df['offsetRa_r']) > 1e3) | (np.abs(df['offsetDec_r']) > 1e3)
    print(df['bastrom_r'][cond_r])
    df['bastrom_r'] = df['bastrom_r'] & ~cond_r
    print(df['bastrom_r'][cond_r])
    print(df[df['bastrom_r'] == True].__len__(), end='\n\n')

    print(df[df['bastrom_i'] == True].__len__())
    cond_i = (np.abs(df['offsetRa_i']) > 1e3) | (np.abs(df['offsetDec_i']) > 1e3)
    print(df['bastrom_i'][cond_i])
    df['bastrom_i'] = df['bastrom_i'] & ~cond_i
    print(df['bastrom_i'][cond_i])
    print(df[df['bastrom_i'] == True].__len__(), end='\n\n')

    print(df[df['bastrom_z'] == True].__len__())
    cond_z = (np.abs(df['offsetRa_z']) > 1e3) | (np.abs(df['offsetDec_z']) > 1e3)
    print(df['bastrom_z'][cond_z])
    df['bastrom_z'] = df['bastrom_z'] & ~cond_z
    print(df['bastrom_z'][cond_z])
    print(df[df['bastrom_z'] == True].__len__(), end='\n\n')

    # print(df)
    df = format_cat(df)
    df.to_csv(f'{path}/init/last/sso_tot4c.csv', index=False)


def checking_KBOs():
    df = pd.read_csv(f'{path}init/last/sso_tot4c.csv')
    l0=df.__len__()
    print(df)
    cond = (np.sqrt(((df['offsetRa_g'] - df['offsetRa_r']) / 0.3) ** 2 + \
                    ((df['offsetDec_g'] - df['offsetDec_r']) / 0.3) ** 2) < 1) & \
           (df['bknown'] == False) | \
           ((df['offsetRa_r'] - df['offsetRa_g']) ** 2 +
            (df['offsetDec_r'] - df['offsetDec_g']) ** 2 > 10000 ** 2)
    slow = df[cond]
    # print(slow[['objID', 'ra', 'dec']])
    # slow.to_csv(f'{path}init/last/slow02.csv')

    df = df[~df['objID'].isin(slow['objID'])].reset_index(drop=True)
    print(l0 - df.__len__())
    df.to_csv(f'{path}init/last/sso_tot4d.csv', index=False)
    # −3 539


def get_sdss_images(df, name, isshow=False):
    def crop_center(img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]

    from scipy.ndimage import rotate
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    n = 15
    for idx in range(0, df.__len__()):
        print(df.iloc[idx]["objID"])
        req1_i = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[idx]["run"]}/{df.iloc[idx]["camcol"]}/' \
                 f'frame-i-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                 f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        req1_r = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[idx]["run"]}/{df.iloc[idx]["camcol"]}/' \
                 f'frame-r-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                 f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        req1_g = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[idx]["run"]}/{df.iloc[idx]["camcol"]}/' \
                 f'frame-g-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                 f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        req1_u = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[idx]["run"]}/{df.iloc[idx]["camcol"]}/' \
                 f'frame-u-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                 f'{df.iloc[idx]["field"]:04d}.fits.bz2'

        fig = plt.figure(figsize=(4, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1, wspace=None, hspace=0.01)

        fname_i = f'frame-i-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                  f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        fname_r = f'frame-r-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                  f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        fname_g = f'frame-g-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                  f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        fname_u = f'frame-u-{df.iloc[idx]["run"]:06d}-{df.iloc[idx]["camcol"]}-' \
                  f'{df.iloc[idx]["field"]:04d}.fits.bz2'
        try:
            if not os.path.exists(f'{path}{name}/fits/{fname_i}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_i, f'{path}{name}/fits/{fname_i}')

            if not os.path.exists(f'{path}{name}/fits/{fname_r}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_r, f'{path}{name}/fits/{fname_r}')

            if not os.path.exists(f'{path}{name}/fits/{fname_g}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_g, f'{path}{name}/fits/{fname_g}')

            if not os.path.exists(f'{path}{name}/fits/{fname_u}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_u, f'{path}{name}/fits/{fname_u}')

            hdu_u = fits.open(f'{path}{name}/fits/{fname_u}')
            w_u = wcs.WCS(hdu_u[0].header)
            world = np.array([[df.iloc[idx]["ra"], df.iloc[idx]["dec"]]])
            pixcrd2 = w_u.wcs_world2pix(world, 0)
            data_u = hdu_u[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]

            hdu_i = fits.open(f'{path}{name}/fits/{fname_i}')
            w_i = wcs.WCS(hdu_i[0].header)
            # world = np.array([[df.iloc[idx]["ra"], df.iloc[idx]["dec"]]])
            world = np.array([[df.iloc[idx]["ra"] + df.iloc[idx]["offsetRa_u"] / 3600.0,
                               df.iloc[idx]["dec"] + df.iloc[idx]["offsetDec_u"] / 3600.0]])
            pixcrd2 = w_i.wcs_world2pix(world, 0)
            data_i = hdu_i[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]

            hdu_r = fits.open(f'{path}{name}/fits/{fname_r}')
            w_r = wcs.WCS(hdu_r[0].header)
            world = np.array([[df.iloc[idx]["ra"] + df.iloc[idx]["offsetRa_u"] / 3600.0,
                               df.iloc[idx]["dec"]+ df.iloc[idx]["offsetDec_u"] / 3600.0]])
            pixcrd2 = w_r.wcs_world2pix(world, 0)
            data_r = hdu_r[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]

            hdu_g = fits.open(f'{path}{name}/fits/{fname_g}')
            w_g = wcs.WCS(hdu_g[0].header)
            # world = np.array([[df.iloc[idx]["ra"], df.iloc[idx]["dec"]]])
            world = np.array([[df.iloc[idx]["ra"] + df.iloc[idx]["offsetRa_u"] / 3600.0,
                               df.iloc[idx]["dec"] + df.iloc[idx]["offsetDec_u"] / 3600.0]])
            pixcrd2 = w_g.wcs_world2pix(world, 0)
            data_g = hdu_g[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]


            if (data_r.shape == (2 * n, 2 * n)) and (data_r.shape == data_g.shape) and (data_r.shape == data_i.shape):
                data = make_lupton_rgb(data_r, data_g*1.3, data_i, Q=10, stretch=0.5)
                # print(data_r.shape, data_i.shape, data_g.shape, data.shape)
                # print(data[:1])
                # data = np.fliplr(data)
                # data = np.rot90(data)
                # if (hdu_g[0].header['CD1_1'] * hdu_g[0].header['CD2_2'] -
                #     hdu_g[0].header['CD1_2'] * hdu_g[0].header['CD2_1']) > 0:
                positionAngle = np.degrees(np.arctan2(hdu_g[0].header['CD1_2'],
                                                      hdu_g[0].header['CD1_1']))
                # else:
                #     positionAngle = -np.degrees(np.arctan2(hdu_g[0].header['CD1_2'],
                #                                           hdu_g[0].header['CD1_1']))

                print('PA = ', positionAngle)
                data = rotate(data, positionAngle)
                if positionAngle < 90:
                    data = data[::-1, :, :]
                # data = crop_center(data, 2 * n, 2 * n)
                # data =
                ax1.imshow(data, origin='lower')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title(df.iloc[idx]['Name'])
            hdu_u.close()
            hdu_g.close()
            hdu_r.close()
            hdu_i.close()
        except:
            print('Strip82')

        X, Y = [], []
        for i, key_i in enumerate(bands.keys()):
            X.append(df.iloc[idx][f'offsetRa_{key_i}'])
            Y.append(df.iloc[idx][f'offsetDec_{key_i}'])
            print(key_i, df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}'])
            # plt.scatter(df.iloc[idx][f'offsetRa_{key_i}'], df.iloc[idx][f'offsetDec_{key_i}'])
            plt.annotate(key_i,
                         xy=(df.iloc[idx][f'offsetRa_{key_i}'] - 0.1,
                             df.iloc[idx][f'offsetDec_{key_i}'] + 0.1))

        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(X, Y)
        # print(slope, intercept, r_value, p_value, slope_std_error)
        print(r_value, p_value, slope_std_error)
        # print(X)
        predict_y = slope * np.array(X) + intercept
        sns.regplot(x=X, y=Y,
                    # line_kws={'label': '$y=%3.7s*x+%3.7s$' % (slope, intercept)}
                    )
        sns.regplot(x=X, y=Y, fit_reg=False, ax=ax2)
        sns.regplot(x=X, y=predict_y, scatter=False, ax=ax2,
                    label=fr'$R^2$ = {r_value**2:.2f}')
        # g_ra = df.iloc[idx][f'offsetRa_g']
        # g_dec = df.iloc[idx][f'offsetDec_g']
        # h_ra = g_ra / 4
        # h_dec = g_dec / 4

        rad2 = 0.4 ** 2
        # plt.plot([0, i_ra, u_ra, z_ra, g_ra], [0, i_dec, u_dec, z_dec, g_dec], 'b.-')
        # ax.add_artist(circle_z)
        # ax.add_artist(circle_u)
        # ax.add_artist(circle_i)
        # plt.plot([i_ra, df.iloc[idx][f'offsetRa_i']], [i_dec, df.iloc[idx][f'offsetDec_i']],
        #          color='Black')
        # plt.plot([u_ra, df.iloc[idx][f'offsetRa_u']], [u_dec, df.iloc[idx][f'offsetDec_u']],
        #          color='Black')
        # plt.plot([z_ra, df.iloc[idx][f'offsetRa_z']], [z_dec, df.iloc[idx][f'offsetDec_z']],
        #          color='Black')

        # plt.scatter([0, i_ra, u_ra, z_ra, g_ra], [0, i_dec, u_dec, z_dec, g_dec], s=100,
        #             edgecolors='Grey', facecolors='none')
        plt.gca().set_aspect('equal')
        ax2.set_xlim(df.iloc[idx][f'offsetRa_g'] + np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetRa_g']),
                 df.iloc[idx][f'offsetRa_r'] - np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetRa_g']))
        ax2.set_ylim(df.iloc[idx][f'offsetDec_g'] + np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetDec_g']),
                 df.iloc[idx][f'offsetDec_r'] - np.sqrt(rad2) * np.sign(df.iloc[idx][f'offsetDec_g']))
        ax2.set_xlabel(r'$\Delta\alpha cos\delta$, arcsec')
        ax2.set_ylabel(r'$\Delta\delta$, arcsec')
        plt.legend()

        if not os.path.exists(f'{path}{name}/fits/figs/{fname_r}/'):
            os.makedirs(f'{path}{name}/fits/figs/{fname_r}/')
        plt.savefig(f'{path}{name}/fits/figs/{fname_r}/{idx:04d}_{fname_r}.png')
        if isshow:
            plt.show()
        else:
            plt.close()


def get_sdss_PS1_images():
    from PIL import Image
    import requests
    from io import BytesIO
    import urllib.request
    import urllib
    # (hypot((offsetRa_g-offsetRa_r)/0.1, (offsetDec_g-offsetDec_r)/0.1) < 1) && (bknown==true)
    # https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/1473/2/frame-g-001473-2-0099.fits.bz2
    # http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?TaskName=Skyserver.Chart.List&ra=157.94981691&dec=-0.60570273&scale=0.2&width=120&height=120&opt=P

    n = 40
    df = pd.read_csv(f'{path}init/last/slow02.csv')
    df = df[~df['ra_sb'].notna()].reset_index()
    print(df)
    for i in range(0, df.__len__()):
    # for i in range(5):
        print(df.iloc[i]["objID"])
        # if df.iloc[i]["objID"] < 1337680046018724228:
        #     continue
        req1_i = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[i]["run"]}/{df.iloc[i]["camcol"]}/' \
                 f'frame-i-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                 f'{df.iloc[i]["field"]:04d}.fits.bz2'
        req1_r = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[i]["run"]}/{df.iloc[i]["camcol"]}/' \
                 f'frame-r-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                 f'{df.iloc[i]["field"]:04d}.fits.bz2'
        req1_g = f'https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/301/' \
                 f'{df.iloc[i]["run"]}/{df.iloc[i]["camcol"]}/' \
                 f'frame-g-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                 f'{df.iloc[i]["field"]:04d}.fits.bz2'
        req2 = f'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?' \
               f'TaskName=Skyserver.Chart.List&' \
               f'ra={df.iloc[i]["ra"]}&' \
               f'dec={df.iloc[i]["dec"]}&' \
               f'scale=0.2&width=120&height=120&opt=P'

        ps1_img = ps1.getcolorim(df.iloc[i]["ra"], df.iloc[i]["dec"],
                                 size=100, output_size=256, filters="gri", format="jpg")

        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        response = requests.get(req2)
        img2 = Image.open(BytesIO(response.content))
        # print(img2)
        ax2.imshow(img2)
        ax2.set_title(f'ra:{df.iloc[i]["ra"]:.4f} dec:{df.iloc[i]["dec"]:.4f}')
        ax3.imshow(ps1_img)
        ax3.set_title(f'PanSTARRS Frame')

        fname_i = f'frame-i-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                  f'{df.iloc[i]["field"]:04d}.fits.bz2'
        fname_r = f'frame-r-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                  f'{df.iloc[i]["field"]:04d}.fits.bz2'
        fname_g = f'frame-r-{df.iloc[i]["run"]:06d}-{df.iloc[i]["camcol"]}-' \
                  f'{df.iloc[i]["field"]:04d}.fits.bz2'

        try:
            if not os.path.exists(f'{path}slow/fits/{fname_i}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_i, f'{path}slow/fits/{fname_i}')

            if not os.path.exists(f'{path}slow/fits/{fname_r}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_r, f'{path}slow/fits/{fname_r}')

            if not os.path.exists(f'{path}slow/fits/{fname_g}'):
                testfile = urllib.request.URLopener()
                testfile.retrieve(req1_g, f'{path}slow/fits/{fname_g}')

            hdu_i = fits.open(f'{path}slow/fits/{fname_i}')
            w_i = wcs.WCS(hdu_i[0].header)
            world = np.array([[df.iloc[i]["ra"], df.iloc[i]["dec"]]])
            # print(world)
            pixcrd2 = w_i.wcs_world2pix(world, 0)
            # print(pixcrd2)
            data_i = hdu_i[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]

            hdu_r = fits.open(f'{path}slow/fits/{fname_r}')
            w_r = wcs.WCS(hdu_r[0].header)
            world = np.array([[df.iloc[i]["ra"], df.iloc[i]["dec"]]])
            # print(world)
            pixcrd2 = w_r.wcs_world2pix(world, 0)
            # print(pixcrd2)
            data_r = hdu_r[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]

            hdu_g = fits.open(f'{path}slow/fits/{fname_g}')
            w_g = wcs.WCS(hdu_g[0].header)
            world = np.array([[df.iloc[i]["ra"], df.iloc[i]["dec"]]])
            pixcrd2 = w_g.wcs_world2pix(world, 0)
            data_g = hdu_g[0].data[int(pixcrd2[0, 1] - n):
                                   int(pixcrd2[0, 1] + n),
                     int(pixcrd2[0, 0] - n):
                     int(pixcrd2[0, 0] + n)]
            if (data_r.shape == (80, 80)) and (data_r.shape == data_g.shape) and (data_r.shape == data_i.shape):
                data = make_lupton_rgb(data_i, data_r, data_g, Q=10, stretch=0.5)
                # print(data_r.shape, data_i.shape, data_g.shape, data.shape)
                # print(data[:1])
                data = np.fliplr(data)
                ax1.imshow(data, origin='lower')
                ax1.set_title(f'SDSS Frame')
        except:
            print('Strip82')


        # data_0 = np.zeros(shape=data_r.shape)
        # data_r = (data_r-data_r.min())/(data_r.max()-data_r.min())
        # data_g = (data_g-data_g.min())/(data_g.max()-data_g.min())
        # print(data_g.shape, data_r.shape, data_0.shape)
        # data = np.dstack([data_r, data_0, data_g])
        # print(data_i[:10], data_r[:10], data_g[:10])

        # plt.savefig(f'{path}slow/fits/figs/known/{i:04d}_{df.iloc[i]["objID"]}.png')
        plt.savefig(f'{path}slow/fits/figs/{i:04d}_{fname_r}.png')
        plt.close()


def tot_group_size():
    df = pd.read_csv(f'{path}init/last/groups/full_groups.csv')
    print(df.__len__())

    df['TAI_r'] = np.where(df['TAI_r'] < 60000, df['TAI_r'] * 24 * 3600, df['TAI_r'])

    df['time'] = round(df['TAI_r'] / 3600)
    df_non = df[df["GroupID"].isna()][:]
    dd = df[["GroupID", "time"]]
    dd = dd.dropna()
    print(dd.info())

    by_groups = dd.groupby(["GroupID", "time"])
    print('Done')
    print(by_groups.size())


def motion_plot():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4d.csv')
    # sdss = sdss[sdss['Name'].notna()]
    # sdss = sdss.sort_values('vel', ascending=False)
    cond = sdss['vel'] < 100
    sdss = sdss[cond]
    # sdss.to_csv(f'{path}init/last/sso_tot4e.csv', index=False)
    #
    # sdss = sdss[sdss['objID']<13e17]
    # sdss = sdss[(sdss['psfMag_r'] < 20) &  (sdss['psfMag_r'] > 18)]
    # cond = (sdss['bastrom_u'] == True) & (sdss['bastrom_i'] == True) & \
    #        (sdss['bastrom_z'] == True)
    # sdss = sdss[cond]
    #             # (sdss['offsetRa_g'] < 20) &  (sdss['offsetDec_g'] > 19)]
    # print(sdss[['vel', 'velErr', 'ra', 'dec', 'psfMag_r']].head(10))
    # get_sdss_images(sdss[20:50])


def get_define_image():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4f.csv')
    # run, camcol, field = 7712, 4, 321
    # run, camcol, field = 8038, 2, 174
    run, camcol, field = 4128, 5, 214
    cond = (sdss['run'] == run) &\
           (sdss['camcol'] == camcol) &\
           (sdss['field'] == field)

    df = sdss[cond]
    col_RA = [col for col in df if col.startswith('offsetRa_')]
    col_DEC = [col for col in df if col.startswith('offsetDec_')]
    X = df[col_RA]
    Y = df[col_DEC]
    r = np.zeros(df.__len__())
    for i in range(df.__len__()):
        _, _, r[i], _, _ = stats.linregress(X.iloc[i], Y.iloc[i])
    df['R2'] = r**2
    # cond1 = (df['R2']>0.95) | (df['bknown']==True)
    # df = df[cond1]
    ltrue = df[df['bknown']==True].__len__()
    l = df.__len__()
    print(l, ltrue)
    # print(df[['bknown', 'psfMag_g', 'psfMag_r', 'psfMag_i']], )
    # if ltrue>l*0.8

    get_sdss_images(df, '7712')


def latitude_distr():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4f.csv', nrows=2000000)
    df = pd.DataFrame(sdss['z'])
    df['r'] = np.sqrt(sdss['x']**2 + sdss['y']**2)
    df['lat'] = np.arctan(sdss['z']/df['r']) * 180 / np.pi
    df['lat'].hist(bins=50, log=True)
    plt.show()


def cat_correction4():
    df = pd.read_csv(f'{path}init/last/sso_tot4e.csv')
    df = df.rename(columns={'class': 'dynclass'})
    cond1 = df['dynclass'].str.startswith('KBO', na=False)
    df.loc[cond1, 'dynclass'] = 'KBO'
    print(df['dynclass'].value_counts())

    col_RA = [col for col in df if col.startswith('offsetRa_')]
    col_DEC = [col for col in df if col.startswith('offsetDec_')]
    X = df[col_RA]
    Y = df[col_DEC]
    r = np.zeros(df.__len__())
    for i in range(df.__len__()):
        if i % 10000 == 0:
            print(i)
        _, _, r[i], _, _ = stats.linregress(X.iloc[i], Y.iloc[i])
    df['R2'] = r ** 2
    df.to_csv(f'{path}init/last/sso_tot4f.csv')


def series_of_asteroids():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4f.csv')
    dr16 = pd.read_csv(f'{path}init/last/field16a_coords.csv')
    dr16 = dr16[dr16['numb'] > 10]
    for index, row in dr16.iterrows():
        print(index)
        cond = (sdss['run'] == row['run']) & (sdss['camcol'] == row['camcol']) &\
        (sdss['field'] == row['field'])
        df = sdss[cond]
        get_sdss_images(df, 'dr16', isshow=False)


def remove_bad_fields():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4f.csv')
    print(sdss)
    k = 6
    stripe82 = pd.read_csv(f'{path}init/last/field82a_coords.csv')
    stripe82 = stripe82[stripe82['numb'] >= k]
    dr16 = pd.read_csv(f'{path}init/last/field16a_coords.csv')
    dr16 = dr16[dr16['numb'] >= k]
    bad_list = []
    bad_frame = []
    for index, row in dr16.iterrows():
        cond = (sdss['run'] == row['run']) & \
               (sdss['rerun'] == row['rerun']) & \
               (sdss['camcol'] == row['camcol']) & \
               (sdss['field'] == row['field'])
        df = sdss[cond]
        if not df['bknown'].any():
            bad_frame.append((row['run'], row['rerun'], row['camcol'], row['field']))
            print(index)
            bad_list.extend(df.index)
            # sdss = sdss.drop(df.index)
            # print(df.index)

    for index, row in stripe82.iterrows():
        cond = (sdss['run'] == row['run']) & \
               (sdss['rerun'] == row['rerun']) & \
               (sdss['camcol'] == row['camcol']) & \
               (sdss['field'] == row['field'])
        df = sdss[cond]
        if not df['bknown'].any():
            print(index)
            bad_frame.append((row['run'], row['rerun'], row['camcol'], row['field']))
            bad_list.extend(df.index)

    sdss = sdss.drop(bad_list)
    print(sdss)
    bad_frames = pd.DataFrame(bad_frame)
    bad_frames.to_csv(f'{path}init/last/bad_frames.csv', index=False)

    sdss.to_csv(f'{path}init/last/sso_tot4g.csv', index=False)



if __name__ == '__main__':
    path = f'/media/gamer/Data/data/sdss/'
    bands = {'r': 0, 'i': 1, 'u': 2, 'z': 3, 'g': 4}
    # get_mjd()
    # load_adr4()
    # check_adr4()
    # check_runs()
    # mysearch()
    # time_diff()

    # fname = '000125-5-0303'
    # sdss_req()
    # show_image()

    # test_adr4_known()
    # concat_tables()

    # skybot_check()
    # skybot_check_known()
    # adr5_analysis()

    # check_run125()
    # concat_adr5()

    # adr5_full()
    # full_sdss_aseroid_check()
    # full_sdss_aseroid_run()
    # concat_sdss()
    # load_mpcorb()

    # sort_ard5c_as_full()
    # adr5d_color()

    # adr5_recover()
    # convert_to_MPC3()
    # adr5_known_clear()
    # adr5_test_coords()
    # catalog_add_fields()
    # clear2()
    # clear3()
    # show_taxonomic()
    # topcat_inter()
    # clear_adr4()
    # clear_panstarrs_known()
    # add_short_name()
    # full_sdss16()

    # adr4_obj_list()
    # adr4i_compare()
    # cas_dr7()
    # adr4i_merge()
    # adr4i_adr4bis_merge()
    # mjd2tai()

    # ps1_math()
    # check_sdr5ps()
    # check_offset()
    # check_mag()
    # check_color()
    # check_known()

    # sdss_query1()
    # sdss_inside()

    # check_neas()

    # Use 3 decimal places in output display
    pd.set_option("display.precision", 12)
    # Don't wrap repr(DataFrame) across additional lines
    # pd.set_option("display.expand_frame_repr", False)
    # Set max rows displayed in output to 25
    # pd.set_option("display.max_rows", 25)
    pd.options.mode.chained_assignment = None  # default='warn'

    # remove_duplicates()
    # concat_csv()
    # clean_tai()
    # skybot_check_tot()
    # concat_by_key()
    # remove_gaia()
    # skybot_remove_duplicates()
    # skybot_analisis()

    # df16 = pd.read_csv(f'{path}/init/last/sso_str82ps1t.csv')
    # df82 = pd.read_csv(f'{path}/init/last/sso_dr16ps1t.csv')
    # a16 = df16['run'].unique()
    # a82 = df82['run'].unique()
    # print(f'len16 = {len(a16)}, len82 = {len(a82)}')
    # eq = set(a16) | set(a82)
    # print(eq, len(eq))
    # calc_r2()
    # adr4_tot3f_compare()
    # concat_numpy()
    # cat_correction1()
    # linarity()
    # matrix_offset()
    # linarity2()
    # taxonomy_show()
    # astrometry_add()

    # cat_correction2()

    # cat_correction3()
    # checking_KBOs()
    # get_sdss_images()

    # sdss = pd.read_csv(f'{path}init/last/sso_tot4d.csv', nrows=20000000)
    # sdss['dynclass'] = sdss['class'].copy()
    # print(sdss['dynclass'].value_counts())
    # column_name = 'dynclass'
    # cond1 = sdss['class'].str.startswith('KBO', na=False)
    # # cond2 = sdss['class'].str.startswith('NEA', na=False)
    # # cond3 = sdss['class'].str.startswith('MB', na=False)
    # sdss.loc[cond1, column_name] = 'KBO'
    # # sdss.loc[cond2, column_name] = 'NEA'
    # # sdss.loc[cond3, column_name] = 'MB'
    # print(sdss['dynclass'].value_counts())

    # for idx, frame in by_groups:
    #     time = frame.iloc[0]['TAI_r']
    #     # frame['diff'] = frame['TAI_r'] - time
    #     frame['status'] = np.abs(frame['TAI_r'] - time) < 60
    #     # frame.loc[np.abs(frame['TAI_r'] - time) < 5, 'status'] = True
    #     new_group = frame.groupby('status')
    #     print(frame[['TAI_r', 'status']])

        # frame['diffs'] = frame.groupby('TAI_r').diff(axis=1)
        # print(frame[['TAI_r', 'diffs']])
        # time_group = frame.groupby('TAI_r')['TAI_r']
        # print(frame['TAI_r'])
        # for _, item in time_group:
        #     print(item)

    # motion_plot()
    # latitude_distr()
    # get_define_image()
    # cat_correction4()
    series_of_asteroids()
    # remove_bad_fields()