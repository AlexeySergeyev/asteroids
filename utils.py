import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from astropy.io import fits
import pandas as pd
from scipy.stats import gaussian_kde
import os
import urllib.request
from astropy import wcs
from astropy.visualization import make_lupton_rgb, simple_norm


def init(isshow=False):
    np.random.seed(0)
    x = 15 + np.random.random(N) * 9
    y = np.random.normal(loc=0.0, scale=0.1, size=N) / (x.max()*1.1 - x)
    ind = np.random.randint(0, N, 5)
    y[ind] = np.random.normal(loc=0.0, scale=10, size=ind.shape[0])
    if isshow:
        plt.scatter(x, y, s=1)
        plt.scatter(x[ind], y[ind], s=10, color='black')
        plt.show()
    return x, y, ind


def seq(isshow=False):
    sx = np.linspace(x.min(), x.max(), sn + 1)
    h2 = (sx[1] - sx[0]) / 2

    med = np.zeros(sn, dtype=float)
    sig = np.zeros(sn, dtype=float)

    sy_data = []
    sx_data = []
    for i in range(sn):
        sy_data.append(y[((x >= sx[i]) & (x < sx[i + 1]))])
        sx_data.append(x[((x >= sx[i]) & (x < sx[i + 1]))])

    for i, sy_seq in enumerate(sy_data[:]):
        med[i] = np.mean(np.array(sy_seq))
        sig[i] = np.std(sy_seq)

    print(sig)

    sy_filtered_data = []
    for i, sy_seq in enumerate(sy_data[:]):
        sy_filtered_data.append(sigma_clip(sy_seq, sigma=3, maxiters=1, cenfunc=np.median))

    # print(sy_filtered_data[0])
    sy_filtered = np.ma.array(sy_filtered_data[0])
    sx_filtered = np.ma.array(sx_data[0])
    for i in range(1, sn):
        sy_filtered = np.ma.concatenate((sy_filtered, sy_filtered_data[i]), axis=None)
        sx_filtered = np.ma.concatenate((sx_filtered, sx_data[i]), axis=None)

    if isshow:
        plt.scatter(x, y, s=1)
        plt.scatter(sx_filtered, sy_filtered, s=1)
        plt.plot(sx[:-1] + h2, med, 'b:')
        plt.errorbar(sx[:-1] + h2, med, sig, fmt='b.', markersize=10)
        plt.scatter(x[ind], y[ind], s=10)
        plt.show()


def sdss_rotate():
    from scipy.ndimage import rotate
    from astropy.visualization.mpl_normalize import ImageNormalize
    from astropy.visualization import SqrtStretch, LogStretch
    from astropy import wcs

    hdu = fits.open('./temp/frame-r-002582-4-0071.fits.bz2')
    data = hdu[0].data
    w = wcs.WCS(hdu[0].header)

    norm = ImageNormalize(stretch=LogStretch())
    plt.imshow(data, origin='lower', cmap='jet', norm=norm)
    plt.show()


def check_tanya():
    sdss = pd.read_csv('/media/gamer/Data/data/sdss/init/last/sso_tot3g.csv')
    nu_sdss = sdss[~sdss["Number"].isna()]
    # nu_sdss['Number'] = nu_sdss['Number'].astype(np.int)
    # nu_sdss['Number'] = nu_sdss['Number'].sort_values()
    # print(nu_sdss)
    tanya = pd.read_csv('./data/very_high-i_MPC.txt')
    df_num = tanya.iloc[:48].astype(np.int)
    df_name = tanya.iloc[48:]
    # print(df_num.info())
    t1 = nu_sdss[nu_sdss['Number'].isin(df_num['name'])]
    print(t1)
    t2 = sdss[sdss['Name'].isin(df_name['name'])]
    print(t2)
    df = pd.concat([t1, t2])
    df.to_csv('./data/tanya.csv')


def ps_image():
    df = pd.read_csv(f'{path}init/gaia_cross.csv')


def test_linear():
    n = 5
    vel = 1
    std = 0.5
    phi = -np.pi/2
    r = np.linspace(0, vel * n, n)
    for phi in np.arange(0, 2*np.pi, 0.2):
        print(phi)
        x = r * np.sin(phi) + np.random.random(n) * std - std/2.0
        y = r * np.cos(phi) + np.random.random(n) * std - std/2.0
        # plt.plot(x, y, marker='o', linestyle='None')
        # plt.plot(r*np.sin(phi), r*np.cos(phi))
        plt.plot(x+y, marker='o')
    plt.grid()
    plt.show()


def test_catalog():
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astropy.coordinates import match_coordinates_sky, match_coordinates_3d

    c1 = SkyCoord(      ra=[1.0, 2.5, 2.0, 1.0] * u.degree,
                       dec=[0.0, 0.0, 0.0, 0.0] * u.degree,
                  distance=[5.0, 0.0, 2.0, 2.5] * u.m)

    c2 = SkyCoord( ra=[1.0, 2.0] * u.degree,
                  dec=[0.0, 0.0] * u.degree,
             distance=[0.0, 0.0] * u.m)

    max_sep = 1.0 * u.deg
    max_sep2 = 1.0 * u.m
    for i in range(len(c1)):
        scalarc = c1[i]
        d2_sky = scalarc.separation(c2)
        d2_3d = scalarc.separation_3d(c2)
        print(d2_sky.deg)
        print(d2_3d)
        cond = np.any(d2_sky < max_sep) & np.any(d2_3d < max_sep2)
        print(cond)
    #

    # idxc1, d2d, d3d = c1.match_to_catalog_3d(c2)
    # # idxc, idxcatalog, d2d, d3d = c1.search_around_sky(c2, max_sep)
    # print(d2d.deg)
    # print(d3d)
    # print(idxc1)
    # # print(idxcatalog)

    # idxc, idxcatalog, d2d, d3d = c1.search_around_3d(c2, max_sep2)
    # print(d2d.deg)
    # print(d3d)
    # print(idxc)
    # print(idxcatalog)

    # sep_constraint = (d2d < max_sep)
    # sep_constraint2 = (d3d < max_sep2)
    # # c_matches = c1[sep_constraint]
    # # print(c_matches)
    # print(sep_constraint)
    # print(sep_constraint2)


def atoiff_proj_plot():
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from scipy import ndimage

    sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv', nrows=2000000)
    deg2rad = np.pi / 180.0
    # ra = (sdss['ra'].to_numpy() - 180) * deg2rad
    # dec = sdss['dec'].to_numpy() / 180 * np.pi
    ra = sdss['ra'].to_numpy()
    dec = sdss['dec'].to_numpy()
    cond = sdss['bknown'] == True
    rak = sdss['ra'][cond].to_numpy()
    deck = sdss['dec'][cond].to_numpy()
    cl = sdss['pn'][cond].to_numpy()
    cond82 = sdss['objID'] > 1337645876861272492
    ra82 = (sdss['ra'][cond82]).to_numpy()
    dec82 = (sdss['dec'][cond82]).to_numpy()

    k = 1
    ra = ra[::k]
    dec = dec[::k]
    rak = rak[::k]
    deck = deck[::k]
    ra82 = ra82[::k]
    dec82 = dec82[::k]

    cl = cl[::k]
    coords = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    # eclip = coords.barycentrictrueecliptic
    # eclip = coords
    coords_k = SkyCoord(ra=rak, dec=deck, unit=u.deg)
    coords_82 = SkyCoord(ra=ra82, dec=dec82, unit=u.deg)

    # print(eclip)

    # color_map = plt.cm.Spectral_r
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='aitoff')
    img = plt.imread("./figs/galaxy1.png")
    print(img.shape)
    img = img[:300, ::-1, :]
    img_c = np.roll(img, img.shape[1] // 2, axis=1)
    print(img_c.shape)
    # img_c = img_c[:, ::-1, :]
    print(img_c.shape[1])
    x = np.linspace(-np.pi, np.pi, img_c.shape[1])
    y = np.linspace(np.pi/2, -np.pi/2 + np.pi/3, img_c.shape[0])
    # y = np.linspace(-np.pi / 2, np.pi / 2, img_c.shape[0])
    # xv, yv = np.meshgrid(x, y)
    # coods_im = SkyCoord(ra=xv * u.deg, dec=yv * u.deg)
    # eclip_im = coods_im.barycentrictrueecliptic
    # .contourf(x, y, z)

    X, Y = np.meshgrid(x, y)
    # plt.imshow(sca)
    plt.pcolormesh(X, Y, img_c[:, :, 2], cmap=plt.cm.Spectral_r)
    # plt.pcolormesh(X, Y, img_c[:, :, 2], cmap=plt.cm.gray_r)

    ax.scatter(coords.ra.wrap_at('180d').radian, coords.dec.radian, s=0.2,  color='red', label='Unknown')
    ax.scatter(coords_k.ra.wrap_at('180d').radian, coords_k.dec.radian, color='Orange', s=0.2, label='Known')
    # ax.scatter(coords_82.ra.radian, coords_82.dec.radian, color='cyan', s=1, label='Strip82')
    ax.scatter(coords_82.ra.wrap_at('180d').radian, coords_82.dec.radian, color='cyan', s=0.2, label='Strip82')

    eclip_lon = np.linspace(179, -179, 100)
    eclip_lat = np.zeros(100)
    eclip = SkyCoord(eclip_lon*u.deg, eclip_lat*u.deg, frame='barycentrictrueecliptic')
    eclip = eclip.icrs
    print(eclip)

    ax.plot(eclip.ra.wrap_at('180d').radian, eclip.dec.radian, color='black')

    # ax.scatter(eclip_im.lon.wrap_at('180d').radian, eclip_im.lat.radian,
    #            s=1, color=img_c[:, :, 2], label='Unknown')
    # image = plt.scatter(rak, deck, s=1, c=cl, cmap=plt.cm.Spectral_r, label='Known')
    # ax.scatter(rak, deck, s=1, label='Known')

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.legend(loc="upper right", bbox_to_anchor=[1.05, 1.15], markerscale=4)
    plt.grid(True)
    ax.set_xticks(np.arange(-np.pi + np.pi/3, np.pi - np.pi/6, np.pi/3))
    ax.set_yticks(np.arange(-np.pi/2 + np.pi/6, np.pi/2 - np.pi/6, np.pi/6))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # for i, val in enumerate(labels):
    #     print(val[:-1])
    #     labels[i] = f'{int(int(val[:-1]) / 15)}h'
    labels = ['16h', '20h', '0h', '4h', '8h']
    ax.set_xticklabels(labels)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig('./figs/aitoff2.png', dpi=300)
    # plt.colorbar(image, spacing='uniform')
    plt.show()


def atoiff_proj_plot2():
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from matplotlib.patches import Rectangle

    sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv', nrows=2000000)
    ra = sdss['ra'].to_numpy()
    dec = sdss['dec'].to_numpy()
    cond82 = sdss['objID'] > 1337645876861272492
    ra82 = (sdss['ra'][cond82]).to_numpy()
    dec82 = (sdss['dec'][cond82]).to_numpy()

    k = 1
    ra = ra[::k]
    dec = dec[::k]
    ra82 = ra82[::k]
    dec82 = dec82[::k]

    coords = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    coords_82 = SkyCoord(ra=ra82, dec=dec82, unit=u.deg)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='aitoff')
    img = plt.imread("./figs/galaxy1.png")
    print(img.shape)
    img = img[:300, ::-1, :]
    img_c = np.roll(img, img.shape[1] // 2, axis=1)
    print(img_c.shape)
    print(img_c.shape[1])
    x = np.linspace(-np.pi, np.pi, img_c.shape[1])
    y = np.linspace(np.pi/2, -np.pi/2 + np.pi/3, img_c.shape[0])

    X, Y = np.meshgrid(x, y)
    # plt.pcolormesh(X, Y, img_c[:, :, 2], cmap=plt.cm.Spectral_r)
    ax.pcolormesh(X, Y, img_c[:, :, 2], cmap=plt.cm.gray_r, label='PS1')
    # extra = Rectangle((0, 0), 1, 1, fc="gray", fill=True, edgecolor='gray', linewidth=1)

    ax.scatter(coords.ra.wrap_at('180d').radian, coords.dec.radian, s=2,  color='Orange', label='DR16')
    ax.scatter(coords_82.ra.wrap_at('180d').radian, coords_82.dec.radian, color='cyan', s=2, label='Strip82')

    eclip_lon = np.linspace(179, -179, 100)
    eclip_lat = np.zeros(100)
    eclip = SkyCoord(eclip_lon*u.deg, eclip_lat*u.deg, frame='barycentrictrueecliptic')
    eclip = eclip.icrs
    print(eclip)

    ax.plot(eclip.ra.wrap_at('180d').radian, eclip.dec.radian, color='black', label='Ecliptic')

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='gray', label='PS1'))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=[1.05, 1.15], markerscale=4,)
    plt.grid(True)
    ax.set_xticks(np.arange(-np.pi + np.pi/3, np.pi - np.pi/6, np.pi/3))
    ax.set_yticks(np.arange(-np.pi/2 + np.pi/6, np.pi/2 - np.pi/6, np.pi/6))
    fig.canvas.draw()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    # for i, val in enumerate(labels):
    #     print(val[:-1])
    #     labels[i] = f'{int(int(val[:-1]) / 15)}h'
    labels = ['16h', '20h', '0h', '4h', '8h']
    ax.set_xticklabels(labels)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    plt.savefig('./figs/aitoff3.png', dpi=300)
    # plt.colorbar(image, spacing='uniform')
    plt.show()


def plot_offset():
    sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv')
    x = sdss['offsetRa_g'] - sdss['offsetRa_r']
    y = sdss['offsetDec_g'] - sdss['offsetDec_r']
    print('Load complete')

    # data = np.vstack([x, y])
    # kde = gaussian_kde(data)
    # print('kde calculated')
    #
    # xgrid = np.linspace(-8, 8, 100)
    # ygrid = np.linspace(-8, 8, 100)
    # Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    # Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    # print('hist calculated')
    #
    # # Plot the result as an image
    # plt.imshow(Z.reshape(Xgrid.shape),
    #            origin='lower', aspect='auto',
    #            extent=[-8, 8, -8, 8],
    #            cmap='Blues')
    # cb = plt.colorbar()
    # cb.set_label("density")
    # plt.show()

    xmin, xmax = -8, 8
    ymin, ymax = -8, 8

    hist, xedges, yedges = np.histogram2d(x, y, bins=400,
               range=[[xmin, xmax], [ymin, ymax]],)
    hist = hist.T

    fig = plt.figure(figsize=(7.5, 6))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    ax1 = fig.add_subplot(111)

    # img = ax1.imshow(np.log10(hist), interpolation='none',
    img=ax1.imshow(hist, interpolation='none',
               origin='low',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='YlOrRd')
    circle1 = plt.Circle((0, 0), radius=0.6, fill=False, color='black', label='')
    circle2 = plt.Circle((0, 0), radius=6.0, fill=False, color='blue', label='')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.set_xlabel(r'$\Delta\alpha cos \delta$ (g-r, arcsec)')
    ax1.set_ylabel(r'$\Delta\delta$ (g-r, arcsec)')
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # cax = inset_axes(ax1,
    #                  width="10%",  # width = 5% of parent_bbox width
    #                  height="400%",  # height : 50%
    #                  loc='right',
    #                  # bbox_to_anchor=(1.05, 0., 1, 1),
    #                  bbox_transform=ax1.transAxes,
    #                  borderpad=0,
    #                )
    cb = plt.colorbar(img, aspect=50)
    cb.set_label("counts")

    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.savefig('./figs/offset2.png', dpi=300)
    plt.show()


    #
    # # x = x[sdss['bknown'] == True]
    # # y = y[sdss['bknown'] == True]
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    # ax2.scatter(x, y, s=1)
    # circle1 = plt.Circle((0, 0), radius=0.6, fill=False, color='red', label='0.6"')
    # circle2 = plt.Circle((0, 0), radius=0.125, fill=False, color='red', label='0.125"/min')
    # ax2.add_patch(circle1)
    # ax2.set_xlabel('offsetRa, "')
    # ax2.set_ylabel('offsetDec, "')
    # ax2.set_xlim(-1, 1)
    # ax2.set_ylim(-1, 1)


def plot_velocity():
    sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv')
    x = sdss['offsetRa_g'] - sdss['offsetRa_r']
    y = sdss['offsetDec_g'] - sdss['offsetDec_r']
    # x = x[sdss['bknown'] == True]
    # y = y[sdss['bknown'] == True]
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.scatter(x, y, s=1)
    circle1 = plt.Circle((0, 0), radius=0.6, fill=False, color='red', label='0.6"')
    circle2 = plt.Circle((0, 0), radius=0.125, fill=False, color='red', label='0.125"/min')
    ax2.add_patch(circle1)
    ax2.set_xlabel('offsetRa, "')
    ax2.set_ylabel('offsetDec, "')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)

    x1 = sdss['rowv'] * 2.5
    y1 = sdss['colv'] * 2.5
    ax1.scatter(x1, y1, s=1)
    ax1.set_xlabel('rowV, "/min')
    ax1.set_ylabel('colV, "/min')
    ax1.add_patch(circle2)

    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    # plt.legend()
    plt.savefig('./figs/offset_example.png')
    plt.show()

def get_stripe82(ser, id, isshow=False):
    subdir = 'stripe82'
    n =15

    def get_file(band):
        fname = f'fpC-{ser["run"]:06d}-{band}{ser["camcol"]}-{ser["field"]:04d}.fit.gz'
        req = f'http://das.sdss.org/imaging/{ser["run"]}/{ser["rerun"]}/corr/{ser["camcol"]}/{fname}'
        # print(req)
        f_out = f'{path}{subdir}/fits/'

        try:
            if not os.path.exists(f_out):
                os.makedirs(f_out)
            if not os.path.exists(f'{f_out}{fname}'):
                print('Downloading')
                testfile = urllib.request.URLopener()
                testfile.retrieve(req, f'{f_out}{fname}')
        except:
            print('Something wrong')

        return fname, f_out

    def get_fits(fname, f_out):
        # print(fname, f_out)
        hdu = fits.open(f_out + fname)
        w = wcs.WCS(hdu[0].header)
        # print(ra, dec)
        world = np.array([[ser["ra"], ser["dec"]]])
        pixcrd2 = w.wcs_world2pix(world, 0)
        # print(pixcrd2)
        data = hdu[0].data[int(pixcrd2[0, 1] - n):
                           int(pixcrd2[0, 1] + n),
                           int(pixcrd2[0, 0] - n):
                           int(pixcrd2[0, 0] + n)]
        positionAngle = np.degrees(np.arctan2(hdu[0].header['CD1_2'],
                                              hdu[0].header['CD1_1']))
        hdu.close()
        return data, positionAngle

    color_data = np.zeros(shape=(2*n, 2*n, 3))
    for idx, band in enumerate(['g', 'r', 'i']):
        fname, file_dir = get_file(band)
        color_data[:, :, idx], pa = get_fits(fname, file_dir)
        med = np.median(color_data[:, :, idx])
        color_data[:, :, idx] -= med
        # color_data[:, :, idx] /= 100

    data = make_lupton_rgb(color_data[:, :, 1],
                           color_data[:, :, 0] * 1.3,
                           color_data[:, :, 2],
                           Q=1, stretch=50)

    fig, ax = plt.subplots()
    ax.imshow(data, origin='lower')
    ax.set_title(ser["Name"])
    ax.set_xticks([])
    ax.set_yticks([])
    if isshow:
        plt.show()
    else:
        fig_out = f'{file_dir}/figs/{ser["run"]:05d}_{ser["camcol"]:01d}_{ser["field"]:04d}/'
        if not os.path.exists(fig_out):
            os.makedirs(fig_out)
        plt.savefig(f'{fig_out}{id:03d}_{fname}.png', dpi=150)
        plt.close()


def load_stripe82_data():
    stripe82 = pd.read_csv(f'{path}init/last/field82a_coords.csv')
    stripe82 = stripe82[stripe82['numb']>10].reset_index()
    stripe82 = stripe82.sort_values('numb')
    print(stripe82)

    sdss = pd.read_csv(f'{path}init/last/sso_tot4f.csv')
    for index, row in stripe82.iterrows():
        # print(row)
        cond = (sdss['run'] == row['run']) & \
               (sdss['rerun'] == row['rerun']) & \
               (sdss['camcol'] == row['camcol']) & \
               (sdss['field'] == row['field'])
        df = sdss[cond].reset_index()
        for i in range(df.__len__()):
            # print(df.iloc[i]['ra'], df.iloc[i]['dec'])
            get_stripe82(df.iloc[i], i, isshow=False)


if __name__ == '__main__':
    path = f'/media/gamer/Data/data/sdss/'
    # N = 10000
    # sn = 20
    # x, y, ind = init(True)
    # seq(True)

    # sdss_rotate()
    # sdss_inside()

    # df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', 'Elephant',
    #                               'Parrot', 'Parrot'],
    #                    'Max Speed': [380., 370., 55., 24., 26.]})
    # by_group = df.groupby(['Animal'])
    # for i, frame in by_group:
    #     print(i, frame)
    # check_tanya()
    # ps_image()
    # test_linear()

    # fig, ax = plt.subplots()
    #
    # # We need to draw the canvas, otherwise the labels won't be positioned and
    # # won't have values yet.
    # fig.canvas.draw()
    #
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # print(labels)
    #
    # ax.set_xticklabels(labels)
    #
    # plt.show()
    # test_catalog()
    # df_tot = pd.read_csv(f'{path}/init/last/sso_tot3f.csv')
    #
    # l = df_tot.__len__()
    # b_adr4 = np.zeros(shape=l, dtype=bool)
    # df_tot['adr4'] = b_adr4
    # cols = df_tot.columns.tolist()
    # print(cols)
    # cols = cols[:1] + cols[21:-11] + cols[1:21] + cols[-11:]
    # print(cols)
    # # df_tot = df_tot[cols]

    # atoiff_proj_plot()
    # atoiff_proj_plot2()

    # plot_velocity()
    # plot_offset()

    load_stripe82_data()