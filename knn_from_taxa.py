#!/usr/bin/env python

import numpy as np
from astropy.io.ascii import read, write
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from astropy.table import join, unique
import pandas as pd
import seaborn as sns


ssize=8
def plot_gr_iz(cols, classed, name):
    plt.scatter(cols[:, 0], cols[:, 2], s=ssize)
    for i, letter in enumerate(complex):
        index = np.where(classed == letter)
        plt.scatter(cols[index, 0],
                    cols[index, 2], label=letter, s=ssize)
    plt.xlabel('g-r')
    plt.ylabel('i-z')
    plt.xlim([0.3, 0.9])
    plt.ylim([-0.6, 0.6])
    plt.legend(loc='lower right')
    # plt.savefig(name)
    plt.show()


def plot_ri_iz(cols, classed, name):
    plt.scatter(cols[:, 1], cols[:, 2], s=ssize)
    for i,letter in enumerate(complex):
        index=np.where(classed == letter)
        plt.scatter(cols[index,1],
                    cols[index,2], label=letter, s=ssize)
    plt.xlabel('r-i')
    plt.ylabel('i-z')
    plt.xlim([0., 0.4])
    plt.ylim([-0.6, 0.6])
    plt.legend(loc='lower right')
    # plt.savefig(name)
    plt.show()


def conf_matr_show(cm, title='', fn='conf_matrix_', issave=False):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=plt.cm.Blues)
    ax.set_xticklabels(d_cls.keys())
    ax.set_yticklabels(d_cls.keys())
    ax.set_title(title)
    if issave:
        plt.savefig(f'./figs/{fn}.png', dpi=150)
    plt.show()

path = f'/media/gamer/Data/data/sdss/'
file_taxa = './data/sso_with_spectra_class.csv'
file_sdss = f'{path}init/last/sso_tot5a.csv'
# sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv', nrows=2000000)
complex = ['A', 'B', 'C', 'D', 'K', 'L', 'Q', 'V', 'S', 'X']
d_cls = {"A": 0, "B": 1, "C": 2, "D": 3, "K": 4, "L": 5, "Q": 6, "V": 7, "S": 8, "X": 9}

taxa = read(file_taxa, delimiter=',')
print(type(taxa))
sdss = read(file_sdss, delimiter=',')
# dps_sdss = pd.read_csv(f'{path}init/last/sso_tot5a.csv', nrows=2000000)
print(type(sdss))

sdss['Name'].name = 'name'

# Simplification of complexes
taxa['complex'][taxa['complex']=='Ch'] = 'C'
taxa['complex'][taxa['complex']=='E'] = 'X'
taxa['complex'][taxa['complex']=='M'] = 'X'
taxa['complex'][taxa['complex']=='O'] = 'S'
taxa['complex'][taxa['complex']=='P'] = 'X'
taxa['complex'][taxa['complex']=='R'] = 'S'

# xmatch sdss with taxonomy
sdssid=sdss[sdss['name'].mask == False ]
sdss_with_taxa = join(sdssid, taxa, keys='name' )


# selecting objects with complex and precise colors
refs = sdss_with_taxa[sdss_with_taxa['g-r'].mask == False ]
refs = refs[refs['r-i'].mask == False ]
refs = refs[refs['i-z'].mask == False ]
refs = refs[refs['psfMagErr_g'] < 0.025]
refs = refs[refs['psfMagErr_r'] < 0.025]
refs = refs[refs['psfMagErr_i'] < 0.025]
refs = refs[refs['psfMagErr_z'] < 0.025]

print(refs)

cols = np.asarray([ refs['g-r'], refs['r-i'], refs['i-z'] ]).T
plot_gr_iz(cols, refs['complex'], 'taxa_gr_iz_learning.png')
plot_ri_iz(cols, refs['complex'], 'taxa_ri_iz_learning.png')
write(refs, 'ml-refs.csv', overwrite=True, delimiter=',')


# Train knn classifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(cols, refs['complex'])
y_pred = neigh.predict(cols)

accuracy = metrics.accuracy_score(refs['complex'], y_pred)
print(accuracy)
cm = metrics.confusion_matrix(refs['complex'], y_pred, normalize=None)
conf_matr_show(cm, title='Spectra: train vs. train', fn='conf_matrix_spectra_1', issave=False)

# select SDSS objects with 3 colors
sdss_full=sdss[ sdss['g-r'].mask == False]
sdss_full=sdss_full[ sdss_full['r-i'].mask == False]
sdss_full=sdss_full[ sdss_full['i-z'].mask == False]
sdsscol_f = np.asarray([sdss_full['g-r'], sdss_full['r-i'], sdss_full['i-z']]).T

# Class all SDSS objects
pred_full = neigh.predict(sdsscol_f)
sdss_full['pc'] = pred_full

plot_gr_iz( sdsscol_f, pred_full, 'taxa_gr_iz_full.png')
plot_ri_iz( sdsscol_f, pred_full, 'taxa_ri_iz_full.png')
write(sdss_full, 'classed_full.csv', overwrite=True, delimiter=',')


# select only those with precise colors
sdss_prec=sdss_full[ sdss_full['psfMagErr_g'] < 0.05]
sdss_prec=sdss_prec[ sdss_prec['psfMagErr_r'] < 0.05]
sdss_prec=sdss_prec[ sdss_prec['psfMagErr_i'] < 0.05]
sdss_prec=sdss_prec[ sdss_prec['psfMagErr_z'] < 0.05]
sdsscol_p = np.asarray( [ sdss_prec['g-r'], sdss_prec['r-i'], sdss_prec['i-z'] ] ).T

# Class all SDSS objects
pred_prec = neigh.predict(sdsscol_p)
sdss_prec['pc'] = pred_prec
plot_gr_iz( sdsscol_p, pred_prec, 'taxa_gr_iz_precise.png')
plot_ri_iz( sdsscol_p, pred_prec, 'taxa_ri_iz_precise.png')
# write(sdss_prec, 'classed_precise.csv', overwrite=True, delimiter=',')



