from astropy_healpix import HEALPix
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def test1():
    NSIDE = 256
    print(
        "Approximate resolution at NSIDE {} is {:.2} deg".format(
            NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
        )
    )
    NPIX = hp.nside2npix(NSIDE)
    print(NPIX)
    m = np.arange(NPIX)
    # print(m)

    # hp.mollview(m, title="Mollview image RING")
    # hp.graticule()
    # plt.show()

    vec = hp.ang2vec(np.pi / 2, 0)
    ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=np.radians(10))
    # m = np.arange(NPIX)
    m[ipix_disc] = m.max()
    print(m)
    # hp.mollview(m, title="Mollview image RING")
    # hp.graticule()

    wmap_map_I = hp.read_map("data/healpix/wmap_band_iqumap_r9_7yr_V_v4.fits")
    # wmap_map_I = hp.read_map("data/healpix/wmap_temperature_analysis_mask_r9_7yr_v4.fits")

    # hp.mollview(
    #     wmap_map_I,
    #     coord=["G", "E"],
    #     title="Histogram equalized Ecliptic",
    #     unit="mK",
    #     norm="hist",
    #     min=-1,
    #     max=1,
    #     cmap='jet'
    # )
    hp.gnomview(wmap_map_I, rot=[0, 0.3], title="GnomView", unit="mK", format="%.2g")

    hp.graticule()
    plt.show()


def test2():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4e.csv', nrows=2000)
    coords = SkyCoord(sdss['ra'], sdss['dec'], unit="deg")

    hp1 = HEALPix(nside=256, frame='icrs')
    print(hp1.pixel_resolution)
    print(hp1.npix)

    healpix_index = hp1.skycoord_to_healpix(coords)
    hp.mollview(healpix_index, title="Mollview image RING")
    plt.show()

    print(healpix_index[0:2])
    print(hp1.nside)


def test3():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4e.csv', nrows=2000000)
    print('Load complete')
    coords = SkyCoord(ra=sdss.ra, dec=sdss.dec, unit="deg")
    NSIDE = 64
    # map = np.zeros(hp.nside2npix(NSIDE))
    print("Approximate resolution at NSIDE {} is {:.2} deg".format(
            NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60))
    print(hp.nside2npix(NSIDE))

    pixel_indices = hp.ang2pix(nside=NSIDE, theta=np.pi - (coords.dec.radian + np.pi / 2),
                               phi=coords.ra.wrap_at('180d').radian)
    print('Indices complete')
    data, _ = np.histogram(pixel_indices, bins=hp.nside2npix(NSIDE), range=(0, hp.nside2npix(NSIDE)))
    map = data
    hp.mollview(map,
                norm="hist",)
    # NPIX = hp.nside2npix(NSIDE)
    plt.show()


def test4():
    sdss = pd.read_csv(f'{path}init/last/sso_tot4e.csv', nrows=2000000)
    sdss= sdss.dropna()
    df = pd.DataFrame(sdss['z'])
    df['r'] = np.sqrt(sdss['x'] ** 2 + sdss['y'] ** 2)
    df['lon'] = np.arctan2(sdss['y'] , sdss['x'])
    df['lat'] = np.arctan(sdss['z'] / df['r'])
    print(df[df['r'].isna()])
    # coords = SkyCoord(ra=df.lon, dec=df.lat, unit=u.rad)
    # print(coords.rad)

    print(np.min(df['lat']), np.max(df['lat']))
    # print(np.min(coords.dec.radian), np.max(coords.dec.radian))
    # print(np.pi - (np.min(df['lat']) + np.pi / 2), np.pi - (np.max(df['lat']) + np.pi / 2))

    NSIDE = 16
    pixel_indices = hp.ang2pix(nside=NSIDE, theta=np.pi - (df['lat'] + np.pi / 2),
                               phi=df['lon'])

    data, _ = np.histogram(pixel_indices, bins=hp.nside2npix(NSIDE), range=(0, hp.nside2npix(NSIDE)))
    map = data
    hp.mollview(map,
                norm="hist", )
    hp.graticule()
    plt.savefig('./figs/lat_lon.png')
    plt.show()


def fields_analysis():
    strip82 = pd.read_csv(f'{path}init/last/field82_coords.csv')
    dr16 = pd.read_csv(f'{path}init/last/field16_coords.csv')
    sdss = pd.read_csv(f'{path}init/last/sso_tot4e.csv')
    dx= 2048 * 0.396
    dy = 1489 * 0.396
    s = dx * dy / (3600**2)
    print(s)

    # strip82['numb'] = np.zeros(strip82.__len__(), dtype=np.int)
    # numb = np.zeros(strip82.__len__(), dtype=np.int)
    # for idx in range(strip82.__len__()):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     cond = (sdss['run'] == strip82.iloc[idx]['run']) & \
    #            (sdss['rerun'] == strip82.iloc[idx]['rerun']) & \
    #            (sdss['camcol'] == strip82.iloc[idx]['camcol']) & \
    #            (sdss['field'] == strip82.iloc[idx]['field'])
    #
    #     temp = sdss[cond]
    #     numb[idx] = temp.__len__()
    # strip82['numb'] = numb
    # print(strip82)
    # strip82.to_csv(f'{path}init/last/field82a_coords.csv', index=False)

    numb = np.zeros(dr16.__len__(), dtype=np.int)
    for idx in range(dr16.__len__()):
        if idx % 1000 == 0:
            print(idx)
        cond = (sdss['run'] == dr16.iloc[idx]['run']) & \
               (sdss['rerun'] == dr16.iloc[idx]['rerun']) & \
               (sdss['camcol'] == dr16.iloc[idx]['camcol']) & \
               (sdss['field'] == dr16.iloc[idx]['field'])

        temp = sdss[cond]
        numb[idx] = temp.__len__()

    dr16['numb'] = numb
    dr16.to_csv(f'{path}init/last/field16a_coords.csv', index=False)


def field_density():
    dx = 2048 * 0.396
    dy = 1489 * 0.396
    s = dx * dy / (3600 ** 2)
    print(s)
    s = 0.0314

    strip82 = pd.read_csv(f'{path}init/last/field82a_coords.csv')
    dr16 = pd.read_csv(f'{path}init/last/field16a_coords.csv')
    strip82 = strip82[strip82['decMax'] > -100]
    df = pd.concat([dr16, strip82])
    coords16 = SkyCoord(df['raMax'], df['decMax'], unit='deg')
    coords16_eclip = coords16.barycentrictrueecliptic
    df['lat'] = coords16_eclip.lat.deg
    print('Calculated ecliptic SDSS')

    ast = pd.read_csv(f'{path}init/asteroids_skybot3d.csv', nrows=1000000)
    coords = SkyCoord(ast['RA'], ast['DEC'], unit='deg')
    coords_eclip = coords.barycentrictrueecliptic
    print('Calculated ecliptic skyBot')

    lat = []
    h = 2
    data, bins = np.histogram(coords_eclip.lat.deg, bins=180//h, range=(-90, 90))
    # print('Calculated ecliptic skyBot')
    for i in range(-90, 90, h):
        cond = (df['lat'] >= i) & (df['lat'] < i+h)
        dd = df[cond]
        lat.append((i, dd['numb'].mean()*np.cos(i/180.0*np.pi)/s, dd['numb'].std()))
        data[(i + 90) // h] = data[(i + 90) // h] / (360 * h) * np.cos(i/180.0*np.pi)

    lat = np.array(lat)
    print(lat)
    # plt.plot(lat[:,0], lat[:,1] , )
    plt.errorbar(lat[:,0]+(h/2), lat[:,1], yerr=lat[:,2], label='SDSS')
    plt.plot(bins[:-1]+(h/2), data, label='Skybot')

    plt.xlabel('Ecliptic latitude')
    plt.ylabel(r'Asteroids number 1 $deg^2$')
    plt.legend()
    plt.savefig('./figs/ast_density.png', dpi=150)
    plt.show()


def test5():
    ast = pd.read_csv(f'{path}init/asteroids_skybot3d.csv', nrows=1000000)
    coords = SkyCoord(ast['RA'], ast['DEC'], unit='deg')
    coords_eclip = coords.barycentrictrueecliptic
    #
    # NSIDE = 64
    # pixel_indices = hp.ang2pix(nside=NSIDE, phi=coords_eclip.lon.wrap_at('180d').radian,
    #                            theta=coords_eclip.lat.radian + np.pi / 2)
    # # pixel_indices = hp.ang2pix(nside=NSIDE, phi=coords.ra.wrap_at('180d').radian,
    # #                            theta=coords.dec.radian + np.pi / 2)
    # data, _ = np.histogram(pixel_indices, bins=hp.nside2npix(NSIDE), range=(0, hp.nside2npix(NSIDE)))
    # map = data
    # hp.mollview(map,
    #             norm="hist", )
    # plt.show()

    hist_lat, bin_edges = np.histogram(coords_eclip.lat, bins=91)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(bin_edges[:-1], hist_lat, width = 2)
    ax.set_yscale('log')
    # plt.xlim(min(bin_edges), max(bin_edges))
    # plt.bar(hist_lat)
    # plt.hist(coords_eclip.lat.deg, bins=90, )
    plt.savefig('./figs/sky3dlat.png')
    plt.show()



    # sdss = pd.read_csv(f'{path}init/last/sso_tot4e.csv', nrows=2000000)
    # sdss = sdss.dropna()
    # df = pd.DataFrame(sdss['z'])
    # df['r'] = np.sqrt(sdss['x'] ** 2 + sdss['y'] ** 2)
    # df['lon'] = np.arctan2(sdss['y'], sdss['x'])
    # df['lat'] = np.arctan(sdss['z'] / df['r'])
    # pixel_indices = hp.ang2pix(nside=NSIDE, theta=np.pi - (df['lat'] + np.pi / 2),
    #                            phi=df['lon'])
    #
    # data, _ = np.histogram(pixel_indices, bins=hp.nside2npix(NSIDE), range=(0, hp.nside2npix(NSIDE)))
    # map = data
    # hp.mollview(map,
    #             norm="jet", )
    # hp.graticule()
    # # plt.savefig('./figs/lat_lon.png')
    # hp.graticule()
    # # plt.savefig('./figs/lat_lon.png')
    plt.show()


if __name__ == '__main__':
    # test1()
    path = f'/media/gamer/Data/data/sdss/'
    # test2()
    # test3()
    # test4()
    # fields_analysis()
    # test5()
    field_density()