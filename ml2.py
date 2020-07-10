import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, neighbors

def simply_class(taxa):
    taxa.loc[taxa['complex'] =='Ch', 'complex'] = 'C'
    taxa.loc[taxa['complex'] == 'E', 'complex'] = 'X'
    taxa.loc[taxa['complex'] == 'M', 'complex'] = 'X'
    taxa.loc[taxa['complex'] == 'O', 'complex'] = 'S'
    taxa.loc[taxa['complex'] == 'P', 'complex'] = 'X'
    taxa.loc[taxa['complex'] == 'R', 'complex'] = 'S'

    return taxa


def prepear_taxo():
    spec1 = pd.read_csv('data/sso_with_spectra_class.csv')
    # print(spec1.info())
    # print(spec1['complex'].value_counts())
    spec1 = simply_class(spec1)
    # print(spec1['complex'].value_counts())
    # print(spec1[spec1['complex'] == 'Ch'])
    df1 = spec1[['name', 'complex']]

    spec2 = pd.read_csv('data/sdss_from_spectra2.csv')
    # print(spec2['complex'].value_counts())

    df = pd.concat([spec2[['name', 'complex']], spec1[['name', 'complex']]])
    # print(df)
    df = df.dropna()
    # print(df)
    print(df['complex'].value_counts())

    df = df.drop_duplicates('name')
    print(df['complex'].value_counts(), df.__len__())
    ast_names = df['name'].to_list

    sdss = pd.read_csv('data/sso_tot4g.csv')
    sdss = sdss.rename(columns={'Name': 'name'})
    spec_sdss = sdss[sdss['name'].isin(df['name'])]

    spec_sdss = pd.merge(spec_sdss, df, left_on='name', right_on='name', how='left')
    print(spec_sdss['complex'].value_counts())

    cond = (spec_sdss['psfMagErr_g'] < 0.05) & (spec_sdss['psfMagErr_r'] < 0.05) & \
           (spec_sdss['psfMagErr_i'] < 0.05) & (spec_sdss['psfMagErr_z'] < 0.05) & \
           (spec_sdss['psfMagErr_u'] < 0.1) & \
           (spec_sdss['bcolor_rg'] == True) & (spec_sdss['bcolor_ri'] == True) & \
           (spec_sdss['bcolor_rz'] == True)

    spec_sdss = spec_sdss[cond]
    print(spec_sdss['complex'].value_counts())
    del spec_sdss['Unnamed: 0']

    spec_sdss.to_csv('data/taxo_ref.csv', index=False)


def plot_gr_iz(df):
    from matplotlib import colors
    fig = plt.figure(figsize=(10, 9))
    ssize = 8
    # plt.scatter(cols[:, 0], cols[:, 2], s=ssize)
    complex_set = set(df['complex'])
    complex = list(complex_set)
    complex.sort()
    for i, letter in enumerate(complex[:]):
        cond = df['complex'] == letter
        temp = ref[cond].reset_index()
        # print(temp)
        plt.scatter(temp['psfMag_r'] - temp['psfMag_i'],
                    temp['psfMag_i'] - temp['psfMag_z'],
                    label=letter, marker=markers[i], s=50,
                    cmap='tab10')
    plt.xlabel('g-r')
    plt.ylabel('i-z')
    # plt.xlim([0.3, 0.9])
    # plt.ylim([-0.6, 0.6])
    plt.legend(loc='lower right')
    # plt.savefig(name)
    plt.show()


def train_knn(refs):
    # Train knn classifier
    cols = np.asarray([refs['g-r'], refs['r-i'], refs['i-z']]).T
    neigh = neighbors.KNeighborsClassifier(n_neighbors=5)
    neigh.fit(cols, refs['complex'])
    y_pred = neigh.predict(cols)


def duplicates_analysis(ref):
    pass

if __name__ == '__main__':
    markers = ['.', '+', 'x', 'v', '^', '1', '2', '3', '4', 's', '*']

    # prepear_taxo()
    ref = pd.read_csv('taxo_ref.csv')
    plot_gr_iz(ref)
    # duplicates_analysis()


    # train_knn(ref)