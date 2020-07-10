import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
# from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
# from sklearn.metrics import confusion_matrix, mean_squared_error, roc_curve, matthews_corrcoef,
# classification_report
import pickle
import xgboost as xgb
from sklearn import datasets


def griz_show(df):
    xedges = np.linspace(0, 1, 100)
    yedges = np.linspace(-0.5, 0.5, 100)

    x = df['psfMag_g'] - df['psfMag_r']
    y = df['psfMag_i'] - df['psfMag_z']

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    plt.imshow(np.log10(H + 1), interpolation='bilinear', origin='low', cmap='jet',
               # norm=LogNorm(vmin=0.1, vmax=10000),
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]]
               )

    cls = pd.read_csv(f'{path}class_colors.csv')
    for i in range(cls.__len__()):
        plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
        plt.annotate(cls.iloc[i]['class'],
                     xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01))

    plt.show()


def color_combinations(df):
    cls = pd.read_csv(f'{path}ml/class_colors.csv')

    key0 = cls.keys()[0]
    print(key0)
    print(df.__len__())

    df_dist = pd.DataFrame()
    for i in range(cls.__len__()):
        # key = cls.keys()[i]
        dist = np.abs(df['psfMag_g'] - df['psfMag_u'] - cls['g-u'][i]) + \
               np.abs(df['psfMag_g'] - df['psfMag_r'] - cls['g-r'][i]) + \
               np.abs(df['psfMag_g'] - df['psfMag_i'] - cls['g-i'][i]) + \
               np.abs(df['psfMag_i'] - df['psfMag_z'] - cls['i-z'][i])
        df_dist[cls['class'][i]] = dist

    min = pd.DataFrame(df_dist.idxmin(axis=1), columns=['class'])
    df['class'] = df_dist.idxmin(axis=1)
    print(min['class'].value_counts())
    # print(df_dist)
    # print(min)
    min['color'] = min['class'].apply(ord) - 65

    # print(min)
    # min['color'] = ord
    # print(cls['class'][min])
    df['u-g'] = df['psfMag_u'] - df['psfMag_g']
    df['u-r'] = df['psfMag_u'] - df['psfMag_r']
    df['u-i'] = df['psfMag_u'] - df['psfMag_i']
    df['u-z'] = df['psfMag_u'] - df['psfMag_z']

    df['g-r'] = df['psfMag_g'] - df['psfMag_r']
    df['g-i'] = df['psfMag_g'] - df['psfMag_i']
    df['g-z'] = df['psfMag_g'] - df['psfMag_z']

    df['r-i'] = df['psfMag_r'] - df['psfMag_i']
    df['r-z'] = df['psfMag_r'] - df['psfMag_z']

    df['i-z'] = df['psfMag_i'] - df['psfMag_z']

    import matplotlib.cm as cm

    # colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    # plt.scatter(df['g-r'], df['i-z'], marker='.', c=min['color'],
    #             cmap='Set3', alpha=0.5, edgecolors='none')

    colors_values = list(set(min['color']))
    colors_names = cls['class'].to_list()
    colors = dict(zip(colors_names, colors_values))
    # print(colors_values, colors_names)
    # for area in [100, 300, 500]:
    #     plt.scatter([], [], c='k', alpha=0.3, s=area,
    #                 label=str(area) + ' km$^2$')
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')

    # cmap = plt.cm.Greys
    # cmin = np.min(colors_values)
    # import matplotlib.colors
    # norm = matplotlib.colors.Normalize(vmin=np.min(colors_values), vmax=np.max(colors_values))
    # for i, val in enumerate(colors_values):
    #     plt.scatter([], [], c=cmap(colors_values[i]), alpha=1,
    #                 label=colors_names[i], norm=norm)
    # plt.legend(scatterpoints=1, frameon=False, labelspacing=1)

    # plt.legend(colors_values, colors_names)
    # plt.legend()

    # for i in range(cls.__len__()):
    #     plt.scatter(cls.iloc[i]['g-r'], cls.iloc[i]['i-z'], marker='.', color='black')
    #     plt.annotate(cls.iloc[i]['class'],
    #                  xy=(cls.iloc[i]['g-r'] + 0.01, cls.iloc[i]['i-z'] + 0.01), weight='bold')
    #     sns.jointplot(x="x", y="y", data=df, kind="kde");

    # N = 11
    # for i in range(N):
    #     sns.palplot(sns.light_palette(sns.color_palette()[0], i+1))
    # print(len(sns.color_palette()))
    # plt.show()
    # for x in df['class'].map(d_cls):
    #     print(x)
    # c = [sns.color_palette()[x] for x in df['class'].map(d_cls)]
    # print(c)
    # plt.show()

    sns.pairplot(df[['u-g', 'u-r', 'u-i', 'u-z',
                     'g-r', 'g-i', 'g-z',
                     'r-i', 'r-z',
                     'i-z', 'class']], hue='class')
    plt.show()


def show_umap(df):
    reducer = umap.UMAP(n_neighbors=50,
                        min_dist=0.01,
                        n_components=4,
                        metric='euclidean')

    new_data = df[['u-g', 'u-r', 'u-i', 'u-z',
                   'g-r', 'g-i', 'g-z',
                   'r-i', 'r-z',
                   'i-z']].values
    scaled_new_data = preprocessing.StandardScaler().fit_transform(new_data)
    embedding = reducer.fit_transform(scaled_new_data)
    print(embedding.shape)

    # for x in df['class'].map(d_cls):
    #     print(x, end=',')
    print(set(df['class']))
    labels = list(set(df['class']))

    a = sns.palplot(sns.color_palette('Spectral', 11))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[sns.color_palette('Paired')[x] for x in df['class'].map(d_cls)],
        marker='.')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Penguin dataset', fontsize=24)
    umap.plot.points(reducer, labels=labels)
    plt.show()


# fig, ax = plt.subplots(1, figsize=(14, 10))
# plt.scatter(*embedding.T, s=0.1, c=target, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])
# cbar = plt.colorbar(boundaries=np.arange(11) - 0.5)
# cbar.set_ticks(np.arange(10))
# cbar.set_ticklabels(classes)
# plt.show()


def load_class_from_spectra():
    df = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    print(df)
    # print(df[df['complex'].isna() == True])
    # print(df['complex'].value_counts())
    # df = df.dropna(subset=['complex'])
    df['complex'] = df['complex'].fillna('U')
    print(df['complex'].value_counts())

    # print(set(df['name']))
    benoit_names = set(df['name'])
    print('asteroids spectra:', len(benoit_names))
    spec = df.groupby('name', as_index=False).first()
    # print(spec)

    sdss = pd.read_csv(f'{path}last/sso_tot4d.csv')
    cond = sdss['bcolor_ru'] & sdss['bcolor_rg'] & sdss['bcolor_ri'] & sdss['bcolor_rz'] & \
           sdss['bastrom_u'] & sdss['bastrom_g'] & sdss['bastrom_r'] & sdss['bastrom_i'] & \
           sdss['bastrom_z'] & sdss['bphot_u'] & sdss['bphot_g'] & sdss['bphot_r'] & \
           sdss['bphot_i'] & sdss['bphot_z']

    sdss = sdss[cond]
    coincidences = sdss[sdss['Name'].isin(benoit_names)]
    print(coincidences)

    sdss_names = list(coincidences['Name'])
    print(len(sdss_names))
    # print(coincidences['Name'])
    print('asteroids sdss:', len(set(coincidences['Name'])))

    classes = np.zeros(coincidences.__len__(), dtype=(np.str))
    ri = np.zeros(coincidences.__len__(), dtype=(np.float))
    rz = np.zeros(coincidences.__len__(), dtype=(np.float))
    iz = np.zeros(coincidences.__len__(), dtype=(np.float))
    for i in range(coincidences.__len__()):
        obj = coincidences.iloc[i]['Name']
        val = spec.loc[spec['name'] == obj]['complex'].values[0]
        classes[i] = val
        ri[i] = spec.loc[spec['name'] == obj]['r-i'].values[0]
        rz[i] = spec.loc[spec['name'] == obj]['r-z'].values[0]
        iz[i] = spec.loc[spec['name'] == obj]['i-z'].values[0]
    coincidences.loc[:, 'complex'] = classes
    coincidences.loc[:, 'max_r-i'] = ri
    coincidences.loc[:, 'max_r-z'] = rz
    coincidences.loc[:, 'max_i-z'] = iz
    # print(coincidences)

    dictarr = []
    for key in d_cls:
        # dictarr.append(d_cls[key])
        dictarr.append(key)

    d_stat = pd.DataFrame(d_cls.items(), columns=['complex', 'id'])
    del d_stat['id']
    # cond = coincidences.loc['class'] == d_stat.loc['class']
    # d_stat.loc['total'] = pd.Series(coincidences[cond].sum())

    coincidences.loc[:, 'u-g'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_g']
    coincidences.loc[:, 'u-r'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[:, 'u-i'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'u-z'] = coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'g-r'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[:, 'g-i'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'g-z'] = coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'r-i'] = coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[:, 'r-z'] = coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[:, 'i-z'] = coincidences.loc[:, 'psfMag_i'] - coincidences.loc[:, 'psfMag_z']

    ast_group = coincidences.groupby('complex')
    print(ast_group['complex'].count())
    d_stat = d_stat.join(ast_group['complex'].count(), on='complex', rsuffix='_r')
    d_stat = d_stat.join(ast_group['u-g'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-r'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['u-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-r'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['g-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['r-i'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['r-z'].mean(), on='complex', rsuffix='_mean')
    d_stat = d_stat.join(ast_group['i-z'].mean(), on='complex', rsuffix='_mean')

    d_stat = d_stat.join(ast_group['u-g'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-r'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['u-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-r'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['g-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['r-i'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['r-z'].std(), on='complex', rsuffix='_std')
    d_stat = d_stat.join(ast_group['i-z'].std(), on='complex', rsuffix='_std')

    # d_stat = d_stat.join(ast_group['u-g'].median(), on='complex', lsuffix='median_')

    # d_stat = d_stat.rename({'u-g_r': ''})
    # d_stat = d_stat.join(ast_group['u-r'].mean(), on='complex', lsuffix='mean_')
    # d_stat = d_stat.join(ast_group['u-r'].median(), on='complex', lsuffix='median_')
    d_stat = d_stat.rename(columns={'complex_r': 'total', 'u-g': 'mean_u-g', 'u-r': 'mean_u-r'})

    pd.set_option("display.precision", 3)
    print(d_stat)
    # # d_stat = pd.merge(d_stat, ast_group['complex'].count(), left_on='complex', right_on='complex')
    # # print(d_stat)
    #
    # tmp = np.zeros(d_stat.__len__())
    # for i in range(d_stat.__len__()):
    #     cond = coincidences['complex'] == d_stat.iloc[i]['complex']
    #     tmp[i] = len(coincidences[cond])
    # d_stat['total'] = tmp
    # print(d_stat)

    print(coincidences['complex'].value_counts())
    isshow = False
    if isshow:
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
        pal = sns.color_palette('Paired', 11)
        cond_c = (coincidences['complex'] == 'A') | (coincidences['complex'] == 'K') | \
                 (coincidences['complex'] == 'L') | (coincidences['complex'] == 'Q') | \
                 (coincidences['complex'] == 'B') | (coincidences['complex'] == 'D')
        cond_c = ~cond_c
        x = coincidences['psfMag_g'][cond_c] - coincidences['psfMag_r'][cond_c]
        y = coincidences['psfMag_i'][cond_c] - coincidences['psfMag_z'][cond_c]
        ax1.scatter(x=x, y=y,
                    c=[pal[x] for x in coincidences['complex'][cond_c].map(d_cls)],
                    marker='.')
        ax2 = sns.barplot(y=dictarr, x=np.ones(len(dictarr)), palette=pal)
        ax2.set_xticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        ax1.set_xlabel('g-r')
        ax1.set_ylabel('i-z')
        ax1.set_xlim([0.1, 1.0])
        ax1.set_ylim([-0.7, 0.7])

        # plt.savefig('./figs/spec_class1.png', dpi=300)
        plt.show()

    issave = False
    if issave:
        coincidences.to_csv(f'{path}ml/coincidences3.csv', index=False)
    # d_stat.to_csv(f'{path}ml/d_stat.csv', index=False)

    return d_stat


def make_combined3():
    spectra = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    spectra = spectra.dropna(subset=['complex']).reset_index()
    spectra_names = set(spectra['name'])
    print('asteroids spectra:', len(spectra_names))
    print('measurements spectra:', spectra.__len__())
    del spectra['class']

    sdss = pd.read_csv(f'{path}last/sso_tot4d.csv')
    coincidences = sdss[sdss['Name'].isin(spectra_names)].reset_index()
    coincidences = coincidences[coincidences['centerdist'] < 2.0]
    print('asteroids sdss:', len(set(coincidences['Name'])))
    print('measurements sdss:', coincidences.__len__())
    # plt.hist(coincidences['centerdist'], bins=100)
    # plt.show()
    # print('asteroids less 3 arcsec', coincidences['centerdist'][coincidences['centerdist'] > 3.0])

    coincidences.loc[(coincidences['bcolor_ru']) & (coincidences['bcolor_rg']), 'u-g'] = \
        coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_g']
    coincidences.loc[coincidences['bcolor_ru'], 'u-r'] = \
        coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[(coincidences['bcolor_ru']) & (coincidences['bcolor_ri']), 'u-i'] = \
        coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[(coincidences['bcolor_ru']) & (coincidences['bcolor_rz']), 'u-z'] = \
        coincidences.loc[:, 'psfMag_u'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[(coincidences['bcolor_rg']), 'g-r'] = \
        coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_r']
    coincidences.loc[(coincidences['bcolor_rg']) & (coincidences['bcolor_ri']), 'g-i'] = \
        coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[(coincidences['bcolor_rg']) & (coincidences['bcolor_rz']), 'g-z'] = \
        coincidences.loc[:, 'psfMag_g'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[(coincidences['bcolor_ri']), 'r-i'] = \
        coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_i']
    coincidences.loc[(coincidences['bcolor_rz']), 'r-z'] = \
        coincidences.loc[:, 'psfMag_r'] - coincidences.loc[:, 'psfMag_z']
    coincidences.loc[(coincidences['bcolor_ri']) & (coincidences['bcolor_rz']), 'i-z'] = \
        coincidences.loc[:, 'psfMag_i'] - coincidences.loc[:, 'psfMag_z']

    sdss_colors = coincidences[['Number', 'Name', 'u-g', 'u-r', 'u-i', 'u-z', 'g-r', 'g-i', 'g-z',
                                'r-i', 'r-z', 'i-z']]

    sdss_colors = sdss_colors.rename(columns={'Number': 'number', 'Name': 'name'})
    # sdss_colors['number'].astype(int)
    classes = np.zeros(coincidences.__len__(), dtype=(np.str))
    for i in range(sdss_colors.__len__()):
        obj = sdss_colors.iloc[i]['name']
        val = spectra.loc[spectra['name'] == obj]['complex'].values[0]
        classes[i] = val
    sdss_colors['complex'] = classes
    # del sdss_colors['index']
    del spectra['index']

    # print(sdss_colors)
    combined = pd.concat([spectra, sdss_colors], ignore_index=True).sort_values('name').reset_index()
    del combined['index']

    cond = combined[color_list].isnull().all(axis=1)
    combined = combined[~cond].reset_index()
    # print(combined[color_list].isnull().all(axis=1))
    for column in color_list:
        combined[column] = combined[column].round(5)

    print(combined)
    combined.to_csv(f'{path}ml/combined2.csv', index=False)


def knn_class():
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score, cross_val_predict

    coincidences = pd.read_csv(f'{path}ml/coincidences.csv')
    coincidences = coincidences[coincidences.complex != 'U']

    X = coincidences[color_list]
    y = coincidences['complex']

    my_scaler = preprocessing.StandardScaler()
    my_scaler.fit(X)
    X_2 = my_scaler.transform(X)

    a_tot = pd.DataFrame(coincidences['complex'].value_counts())
    a_tot = a_tot.rename(columns={'complex': 'total'})
    print(a_tot)

    for state in range(0, 1):
        X_train, X_test, y_train, y_test = \
            train_test_split(X_2, y, stratify=y, random_state=state)

        all_scores = []
        n_neighb = np.arange(10, 11)
        for n_neighbors in n_neighb:
            # print(n_neighbors)
            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
            knn_clf.fit(X_train, y_train)
            # knn_clf.score(X_train, y_train)
            # print(knn_clf.score(X_train, y_train))
            knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=3)

            Z = cross_val_predict(knn_clf, X, y, cv=3)
            cond = (Z != y)

            all_scores.append((n_neighbors,
                               knn_scores.mean(),
                               knn_scores.std(),
                               coincidences.loc[:, ['complex']][cond].__len__()
                               ))
            wrong = coincidences.loc[:, ['complex']][cond]
            a1 = pd.DataFrame(wrong['complex'].value_counts())
            # a_tot['wrong'] = a1['complex']
            # print((a_tot - a1)/ a_tot)
            a_tot = pd.merge(a_tot, a1, left_index=True, right_index=True)
            a_tot = a_tot.rename(columns={'complex': 'wrong class'})
            # print(a1)
            print(a_tot)
        # print(coincidences['complex'].value_counts())

        coincidences['predict'] = Z
        # res = np.array(sorted(all_scores, key=lambda x: x[1], reverse=True))
        all_scores = np.array(all_scores)
        # for item in res:
        #     print(item)
        # Z = knn_clf.predict(X)
        # cond = (Z != y)
        # df_wrong = coincidences.loc[:, ['complex']][cond]
        # print(df_wrong.__len__())

        # my_scaler = preprocessing.StandardScaler()
        # my_scaler.fit(X)
        # X_2 = my_scaler.transform(X)
        # # X_2 = preprocessing.scale(X)
        # print(X_2.std(axis=0))
        #
        # neighb = np.arange(2, 50)
        # # print(neighb)
        # score1 = np.zeros(neighb.shape[0])
        # for i, n_neighbors in enumerate(neighb):
        #     # we create an instance of Neighbours Classifier and fit the data.
        #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform',n_jobs=4)
        #     clf.fit(X, y)
        #     Z = clf.predict(X)
        #     cond = (Z != y)
        #     df_wrong = coincidences.loc[:, ['complex']][cond]
        #     score1[i] = df_wrong.__len__()
        #     print(i, end=', ')
        #
        # print()
        # score2 = np.zeros(neighb.shape[0])
        # for i, n_neighbors in enumerate(neighb):
        #     # we create an instance of Neighbours Classifier and fit the data.
        #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform', n_jobs=4)
        #     clf.fit(X_2, y)
        #     Z = clf.predict(X_2)
        #     cond = (Z != y)
        #     df_wrong = coincidences.loc[:, ['complex']][cond]
        #     score2[i] = df_wrong.__len__()
        #     print(i, end=', ')
        #
        isshow = False
        if isshow:
            plt.ylabel('score')
            plt.xlabel('neighbors')

            # plt.plot(n_neighb, all_scores[:, 1], label='X')
            plt.plot(n_neighb, all_scores[:, 1], label=f'n_neighb')
            plt.errorbar(x=n_neighb, y=all_scores[:, 1], yerr=all_scores[:, 2])
        # plt.savefig('./figs/knn_score.png', dpi=150)

        isshow = True
        if isshow:
            dictarr = []
            for key in d_cls:
                # dictarr.append(d_cls[key])
                dictarr.append(key)
            fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]})
            pal = sns.color_palette('Paired', 11)
            # cond_c = (coincidences['complex'] == 'A') | (coincidences['complex'] == 'K') | \
            #          (coincidences['complex'] == 'L') | (coincidences['complex'] == 'Q') | \
            #          (coincidences['complex'] == 'B') | (coincidences['complex'] == 'D')
            cond_c = (coincidences['predict'] == 'A') | (coincidences['predict'] == 'K') | \
                     (coincidences['predict'] == 'L') | (coincidences['predict'] == 'Q') | \
                     (coincidences['predict'] == 'B') | (coincidences['predict'] == 'D')
            cond_c = cond_c
            x = coincidences['psfMag_g'][cond_c] - coincidences['psfMag_r'][cond_c]
            y = coincidences['psfMag_i'][cond_c] - coincidences['psfMag_z'][cond_c]
            ax1.scatter(x=x, y=y,
                        c=[pal[x] for x in coincidences['predict'][cond_c].map(d_cls)],
                        marker='.')
            ax2 = sns.barplot(y=dictarr, x=np.ones(len(dictarr)), palette=pal)
            ax2.set_xticks([])
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)

            ax1.set_xlabel('g-r')
            ax1.set_ylabel('i-z')
            ax1.set_xlim([0.1, 1.0])
            ax1.set_ylim([-0.7, 0.7])
            ax1.set_title('A, B, D, K, L, Q classes')
            # ax1.set_title('C, S, V, X classes')
            plt.savefig('./figs/predict_class1.png', dpi=150)
        # plt.show()

    # plt.legend()
    plt.show()


#
# # cond = (Z != y)
# # print(coincidences.loc[:, ['complex']][cond])
# # print(test_score)


def pca_analysis():
    #  principal component analysis PCA
    from sklearn import decomposition

    coincidences = pd.read_csv(f'{path}ml/coincidences.csv')
    coincidences = coincidences[coincidences.complex != 'U']

    X = coincidences[color_list]
    y = coincidences['complex']

    pca = decomposition.PCA()
    ast_pca = pca.fit_transform(X)
    print(ast_pca.shape)
    var = pca.explained_variance_ratio_
    print(var)


def conf_matr_show(cm, title='', fn='conf_matrix_', issave=False):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=plt.cm.Blues)
    ax.set_xticklabels(d_cls.keys())
    ax.set_yticklabels(d_cls.keys())
    ax.set_title(title)
    if issave:
        plt.savefig(f'./figs/{fn}.png', dpi=150)
    plt.show()


def max_analysis():
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    import matplotlib.cm as cmp

    issave = False
    # isnorm = None
    isnorm = 'true'
    df = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    # print(df)
    df_good = df.dropna(subset=['r-i', 'i-z', 'r-z', 'complex'])
    print(df_good['complex'].value_counts())
    print(df_good.__len__())

    X = df_good[['r-i', 'i-z', 'r-z']]
    y = df_good[['complex']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, random_state=0)
    knn_clf = neighbors.KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(2, 20, 1))}

    gs = GridSearchCV(knn_clf, param_grid, cv=4)
    gs.fit(X_train, y_train.values.ravel())
    print(gs.best_params_)
    # print(gs.cv_results_['mean_test_score'])

    # a = zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
    # for item in a:
    #     print(item)

    # param_dist = param_grid
    # rs = RandomizedSearchCV(knn_clf, param_dist, cv=10, n_iter=18)
    # rs.fit(X_train, y_train.values.ravel())

    # print(rs.best_params_)
    # a = zip(rs.cv_results_['params'], rs.cv_results_['mean_test_score'])
    # for item in a:
    #     print(item)

    y_pred = gs.predict(X)

    cm = metrics.confusion_matrix(y, y_pred, normalize=isnorm)
    # print(cm)
    print(np.diagonal(cm).sum(), np.sum(cm), np.diagonal(cm).sum() / np.sum(cm))
    print(gs.best_score_)
    conf_matr_show(cm, title='Spectra: train vs. test', fn='conf_matrix_spectra_1', issave=issave)

    # print(rs.best_score_)

    # for n_neighbors in range(2, 10):
    #     clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    #     clf.fit(X_train, y_train)
    #     all_scores.append((n_neighbors, cross_val_score(clf, X_train,
    #                                                     y_train, cv=4).mean()))
    #     y_test = clf.predict(X_train)
    #     print(y_test)
    #     print(y_train)
    #     y_test = cross_val_predict(clf, X_train, y_train, cv=3)
    #     cond = (y_test != y_train)
    #     wrong = df_good.loc[:, ['complex']][cond]
    #
    #     a1 = pd.DataFrame(wrong['complex'].value_counts())
    #     # a_tot['wrong'] = a1['complex']
    #     # print((a_tot - a1)/ a_tot)
    #     a_tot = pd.merge(a_tot, a1, left_index=True, right_index=True)
    #     a_tot = a_tot.rename(columns={'complex': 'wrong class'})
    #     # df_wrong = df_good.loc[:, ['complex']][cond]
    #     # score2[n_neighbors-2] = df_wrong.__len__()
    #     # print(n_neighbors, end=', ')

    # my_scaler = preprocessing.StandardScaler()
    # my_scaler.fit(X)
    # X_2 = my_scaler.transform(X)

    # df_gr = df.groupby('name')
    # for color in color_list:
    #     cond = df[df.groupby(['name'])[color].transform('count') > 0]
    #     print(f'Color {color}: {cond.__len__()}')

    # X = coincidences[color_list]
    # y = coincidences['complex']

    coinc_init = pd.read_csv(f'{path}ml/coincidences2.csv')
    coinc = coinc_init[coinc_init.complex != 'U']

    X2 = coinc[['r-i', 'i-z', 'r-z']]
    y2 = coinc[['complex']]

    y2_pred = gs.predict(X2)
    cm2 = metrics.confusion_matrix(y2, y2_pred, normalize=isnorm)
    # print(cm2)
    print(np.diagonal(cm2).sum(), np.sum(cm2), np.diagonal(cm2).sum() / np.sum(cm2))
    conf_matr_show(cm2, title='Spectra train vs. SDSS colors', fn='conf_matrix_SDSS_1', issave=issave)

    X3 = coinc[['r-i', 'i-z', 'r-z']]
    y3 = coinc[['complex']]

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3,
                                                            stratify=y3, random_state=0)
    knn_clf3 = neighbors.KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(2, 20, 1))}

    gs3 = GridSearchCV(knn_clf3, param_grid, cv=4)
    gs3.fit(X3_train, y3_train.values.ravel())
    print(gs3.best_params_)

    y3_pred = gs3.predict(X3)
    cm3 = metrics.confusion_matrix(y3, y3_pred, normalize=isnorm)
    # print(cm3)
    print(np.diagonal(cm3).sum(), np.sum(cm3), np.diagonal(cm3).sum() / np.sum(cm3))
    conf_matr_show(cm3, title='SDSS train vs. SDSS test', fn='conf_matrix_SDSS_2', issave=issave)

    X4 = coinc[color_list]
    y4 = coinc[['complex']]

    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4,
                                                            stratify=y4, random_state=0)
    knn_clf4 = neighbors.KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(2, 20, 1))}

    gs4 = GridSearchCV(knn_clf4, param_grid, cv=4)
    gs4.fit(X4_train, y4_train.values.ravel())
    print(gs4.best_params_)

    y4_pred = gs4.predict(X4)
    cm4 = metrics.confusion_matrix(y4, y4_pred, normalize=isnorm)
    # print(cm3)
    print(np.diagonal(cm4).sum(), np.sum(cm4), np.diagonal(cm4).sum() / np.sum(cm4))
    conf_matr_show(cm4, title='SDSS train vs. SDSS test', fn='conf_matrix_SDSS_3', issave=issave)


# y4_pred = gs3.predict(X4)
#
# # colors = cm.rainbow(np.linspace(0, 1, len(d_cls[]))))
# colors = cmp.rainbow(np.linspace(0, 1, coinc_init.__len__()))
#
# for i in range(coinc_init.__len__()):
#     color = coinc_init.iloc[i][['complex']]
#     plt.scatter(coinc_init.iloc[i]['g-r'], coinc_init.iloc[i]['i-z'],
#             edgecolor=colors[i],
#             facecolor='none',
#             marker='.')
# plt.show()


def max_me_phot():
    df_max = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    df_max = df_max.dropna(subset=['complex'])

    df = pd.read_csv(f'{path}ml/coincidences2.csv')
    df = df[df.complex != 'U']
    isshow = True
    if isshow == True:
        plt.scatter(df['max_r-i'], df['r-i'], marker='.', label='r-i')
        plt.scatter(df['max_i-z'], df['i-z'], marker='.', label='i-z')
        plt.scatter(df['max_r-z'], df['r-z'], marker='.', label='r-z')
        plt.plot([-0.4, 0.4], [-0.4, 0.4], color='black')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Spectra color')
        plt.ylabel('SDSS color')
        plt.savefig('./figs/max_sdss_colors.png', dpi=150)
        plt.show()

    if isshow == True:
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(df['r-i'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='r-i')
        ax1.hist(df['i-z'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='i-z')
        ax1.hist(df['r-z'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='r-z')
        ax1.set_title('SDSS colors')
        ax1.legend()
        # plt.show()
        ax2.hist(df_max['r-i'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='r-i')
        ax2.hist(df_max['i-z'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='i-z')
        ax2.hist(df_max['r-z'], alpha=0.5, bins=20, range=(-0.4, 0.4), label='r-z')
        ax2.set_title('Spectra colors')
        ax2.legend()
        plt.savefig('./figs/max_sdss_colors_hist.png', dpi=150)
        plt.show()


def my_knn_analysis():
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import confusion_matrix

    df = pd.read_csv(f'{path}ml/coincidences.csv')
    df = df[df.complex != 'U']
    print(df)

    X = df[['r-i', 'i-z', 'r-z']]
    y = df['complex']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y)
    knn_clf = neighbors.KNeighborsClassifier()
    param_grid = {'n_neighbors': list(range(2, 20, 1))}

    gs = GridSearchCV(knn_clf, param_grid, cv=4)
    gs.fit(X_train, y_train.values.ravel())
    print(gs.best_params_)
    # print(gs.cv_results_['mean_test_score'])

    # a = zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score'])
    # for item in a:
    #     print(item)

    # param_dist = param_grid
    # rs = RandomizedSearchCV(knn_clf, param_dist, cv=10, n_iter=50)
    # rs.fit(X_train, y_train.values.ravel())
    #
    # print(rs.best_params_)
    # a = zip(rs.cv_results_['params'], rs.cv_results_['mean_test_score'])
    # for item in a:
    #     print(item)

    y_pred = gs.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print(gs.best_score_)


# print(rs.best_score_)


def feature_import():
    from sklearn.ensemble import ExtraTreesClassifier
    coincidences = pd.read_csv(f'{path}ml/coincidences.csv')
    coincidences_U = coincidences[coincidences.complex == 'U']
    coincidences = coincidences[coincidences.complex != 'U']

    X = coincidences[color_list]
    y = coincidences['complex']

    model = ExtraTreesClassifier()
    model.fit(X, y)
    # cmap =
    for i in range(len(color_list)):
        print(f'{color_list[i]}: {model.feature_importances_[i]:.2f}')
    # plt.scatter(color_list, model.feature_importances_, color=i, cmap='Set3')

    Z = coincidences_U[color_list]
    z = pd.DataFrame(model.predict(Z), columns=['p_complex'])
    # cond = (y == z)

    print(z['p_complex'].value_counts())
    # color_ind = color_list.index()
    # print(color_ind)
    pal = sns.color_palette('Paired', 11)
    sns.barplot(y=model.feature_importances_, x=color_list, palette=pal)
    # print(model.feature_importances_)
    # plt.scatter(color_list, model.feature_importances_, color=color_list.index())
    plt.show()


def taxonomy_show():
    from matplotlib.patches import Ellipse
    import matplotlib.cm as cm

    coin = pd.read_csv(f'{path}ml/coincidences.csv')
    df = pd.read_csv(f'{path}ml/class_colors.csv')
    d_stat = pd.read_csv(f'{path}ml/d_stat.csv')
    print(df)
    print(d_stat)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    colors = cm.rainbow(np.linspace(0, 1, df.__len__()))
    for i in range(df.__len__()):
        x = df.iloc[i]['g-r']
        y = df.iloc[i]['i-z']
        sx = df.iloc[i]['sigma g-r']
        sy = df.iloc[i]['sigma i-z']
        # print(x, y, sx, sy)
        ellipse1 = Ellipse(xy=(x, y),
                           width=sx,
                           height=sy,
                           edgecolor=colors[i],
                           facecolor='None',
                           lw=1, linestyle='--', label=f'{df.iloc[i]["class"]} DM_C')
        ax.add_patch(ellipse1)
        ellipse2 = Ellipse(xy=(d_stat.iloc[i]['g-r'], d_stat.iloc[i]['i-z']),
                           width=d_stat.iloc[i]['g-r_std'],
                           height=d_stat.iloc[i]['i-z_std'],
                           edgecolor=colors[i],
                           facecolor='None',
                           lw=1, label=f'{d_stat.iloc[i]["complex"]} SDSS')
        ax.add_patch(ellipse2)
        plt.scatter(x, y, edgecolor=colors[i], facecolor=colors[i], marker='o')
        plt.annotate(f'{df.iloc[i]["class"]}', xy=(x, y))
        plt.scatter(d_stat.iloc[i]['g-r'], d_stat.iloc[i]['i-z'], edgecolor=colors[i], facecolor=colors[i], marker='x')
        plt.annotate(f'{d_stat.iloc[i]["complex"]}', xy=(d_stat.iloc[i]['g-r'], d_stat.iloc[i]['i-z']))

    for i in range(coin.__len__()):
        color = coin.iloc[i]['complex']
        if color != 'U':
            # print(color)
            # print(d_cls[color])
            plt.scatter(coin.iloc[i]['g-r'], coin.iloc[i]['i-z'],
                        edgecolor=colors[d_cls[color]],
                        # facecolor=colors[d_cls[color]],
                        facecolor='none',
                        marker='.')

        # circle = plt.((df.iloc[i]['g-u'], df.iloc[i]['i-z']), )

    # plt.scatter(coinc['g-r'], coinc['i-z'], marker='.')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.xlim(0.3, 1)
    plt.ylim(-0.4, 0.3)
    plt.savefig('./figs/class_small.png', dpi=300)
    plt.show()


def combining_knn():
    coinc_init = pd.read_csv(f'{path}ml/coincidences2.csv')
    coinc = coinc_init[coinc_init.complex != 'U']
    spectra_init = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    spectra = spectra_init.dropna(subset=['r-i', 'i-z', 'r-z', 'complex'])

    combined = spectra[['r-i', 'i-z', 'r-z', 'complex', 'name']]
    combined = combined.rename(columns={'name': 'Name'})
    combined = combined.append(coinc[['r-i', 'i-z', 'r-z', 'complex', 'Name']], ignore_index=True)

    coinc_set = set(coinc['Name'])
    spectra_set = set(spectra['name'])
    print(len(coinc_set))
    print(len(spectra_set))
    merge = coinc_set.union(spectra_set)
    print(len(merge))
    combined_set = set(combined['Name'])
    print(len(combined_set))
    print(combined['complex'].value_counts())
    combined = combined[combined['Name'].isin(combined_set)]
    print(combined)

    X = combined[['r-i', 'i-z', 'r-z']]
    y_init = combined[['complex']]
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y_init)
    y = y_init.values.ravel()
    print(set(y))

    mean = []
    std = []
    n_neib = np.arange(11, 12, 2)
    for j in n_neib:
        data = []

        for i in range(1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                stratify=y,
                                                                # random_state=0,
                                                                )
            scaller = preprocessing.StandardScaler()
            scaller.fit(X_train)
            X_train = scaller.transform(X_train)
            X_test = scaller.transform(X_test)

            knn_clf = neighbors.KNeighborsClassifier(n_neighbors=j)

            knn_clf.fit(X_train, y_train)
            y_pred = knn_clf.predict(X_test)

            data.append(metrics.accuracy_score(y_test, y_pred))
        # print(metrics.accuracy_score(y_test, y_pred))
        t1 = np.mean(data)
        mean.append(t1)
        t2 = np.std(data)
        std.append(t2)
        print(j, t1, t2)

    y_pred = knn_clf.predict(X)
    cm = metrics.confusion_matrix(y, y_pred, normalize='true')
    # print(cm3)
    # print(np.diagonal(cm4).sum(), np.sum(cm4), np.diagonal(cm4).sum() / np.sum(cm4))
    conf_matr_show(cm, title='SDSS+Spectra', fn='conf_matrix_SDSS_3')

    # print(mean, std)
    # plt.hist(data)
    plt.plot(n_neib, mean)
    plt.ylim(0, 1)
    plt.errorbar(n_neib, mean, yerr=std)
    plt.show()


def knn2():
    combined = pd.read_csv(f'{path}ml/combined.csv')

    X = combined[['r-i', 'i-z', 'r-z']]
    y_init = combined[['complex']]
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y_init)
    y = y_init.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=0,
                                                        )
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=11)
    knn_clf.fit(X_train, y_train)

    y_pred = knn_clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(acc)

    cm = metrics.confusion_matrix(y_test, y_pred, normalize=None)
    print(np.diagonal(cm).sum(), np.sum(cm), np.diagonal(cm).sum() / np.sum(cm))
    conf_matr_show(cm, title='SDSS+Spectra', fn='conf_matrix_combined', issave=False)


def save_combined():
    coinc_init = pd.read_csv(f'{path}ml/coincidences2.csv')
    coinc = coinc_init[coinc_init.complex != 'U']
    # print(coinc)
    print(coinc['Name'])
    spectra_init = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
    spectra = spectra_init.dropna(subset=['r-i', 'i-z', 'r-z', 'complex'])

    combined = coinc[color_list]
    combined.loc[:, 'complex'] = coinc.loc[:, 'complex']
    combined.loc[:, 'name'] = coinc.loc[:, 'Name']
    combined = combined.append(spectra[['r-i', 'i-z', 'r-z', 'complex', 'name']], ignore_index=True)
    print(combined['complex'].value_counts())


# combined.to_csv(f'{path}ml/combined.csv', index=False)
# print(combined)

# coinc_init = pd.read_csv(f'{path}ml/coincidences2.csv')
# coinc = coinc_init[coinc_init.complex != 'U']
# spectra_init = pd.read_csv(f'{path}ml/sdss_from_spectra.csv')
# spectra = spectra_init.dropna(subset=['r-i', 'i-z', 'r-z', 'complex'])
#
# combined = spectra[['r-i', 'i-z', 'r-z', 'complex', 'name']]
# combined = combined.rename(columns={'name': 'Name'})
# combined = combined.append(coinc[['r-i', 'i-z', 'r-z', 'complex', 'Name']], ignore_index=True)
#
# coinc_set = set(coinc['Name'])
# spectra_set = set(spectra['name'])
# print(len(coinc_set))
# print(len(spectra_set))
# merge = coinc_set.union(spectra_set)
# print(len(merge))
# combined_set = set(combined['Name'])
# print(len(combined_set))
# print(combined['complex'].value_counts())
# combined = combined[combined['Name'].isin(combined_set)]
# print(combined)


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


def xgboost_test():
    def display_scores(scores):
        print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

    def auc(m, train, test):
        return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]),
                metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))

    def auc2(selected_classifier, train_set_dataframe, test_set_dataframe, train_class):
        roc = {label: [] for label in y.unique()}
        for label in y.unique():
            selected_classifier.fit(train_set_dataframe, train_class == label)
            predictions_proba = selected_classifier.predict_proba(test_set_dataframe)
            roc[label] += metrics.roc_auc_score(y, predictions_proba[:, 1])

    from xgboost import XGBClassifier
    combined = pd.read_csv(f'{path}ml/combined2.csv')
    print(combined)

    # X = combined[['r-i', 'i-z', 'r-z']].to_numpy()
    # print(X)
    X = combined[color_list].to_numpy()
    y_init = combined[['complex']]
    n_classes = len(set(y_init))
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y_init)
    y = y.ravel()

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=0)

    model = XGBClassifier(gpu_id=0)
    param_dist = {"max_depth": [3, 5, 10],
                  "min_child_weight": [1, 3, 5, 7],
                  "n_estimators": [200],
                  "learning_rate": [0.02, 0.05, 0.1, 0.2]}

    grid_search = GridSearchCV(model, param_grid=param_dist, cv=4,
                               verbose=10, n_jobs=-1)

    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # with open(f'{path}ml/xb2_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    # auc(model, X_train, X_test)

    # # make predictions for test data
    # y_pred = model.predict(X_test)
    # scores.append(metrics.accuracy_score(y_test, y_pred))
    # # y_pred = model.predict(X)
    # predictions = [round(value) for value in y_pred]
    # # evaluate predictions
    # accuracy = metrics.accuracy_score(y_test, predictions)
    # # accuracy = metrics.accuracy_score(y, predictions)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #

    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')
    print(metrics.accuracy_score(y_test, y_pred))
    print(np.diagonal(cm).sum(), np.sum(cm), np.diagonal(cm).sum() / np.sum(cm))
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')
    print(np.diagonal(cm).sum(), np.sum(cm), np.diagonal(cm).sum() / np.sum(cm))
    conf_matr_show(cm, title='XGBoost', fn='conf_matrix_xgb_true', issave=True)


# display_scores(np.sqrt(scores))
# Compute ROC curve and ROC area for each class

# y_score = metrics.roc_auc_score(y_test, y_pred)
# print(y_score)


def knn_hyper1():
    combined = pd.read_csv(f'{path}ml/combined.csv')

    X = combined[['r-i', 'i-z', 'r-z']]
    y_init = combined[['complex']]
    # le = preprocessing.LabelEncoder()
    # y = le.fit_transform(y_init)
    y = y_init.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y,
                                                        random_state=0,
                                                        )

    model = neighbors.KNeighborsClassifier(n_jobs=-1)
    # Hyper Parameters Set
    params = {'n_neighbors': np.arange(3, 21, 2),
              'leaf_size': [1, 2, 3, 5],
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'n_jobs': [-1]}
    print('Start')
    # Making models with hyper parameters sets
    model1 = GridSearchCV(model, param_grid=params, cv=4,
                          verbose=10, n_jobs=-1)
    # Learning
    model1.fit(X_train, y_train)
    # The best hyper parameters set
    print("Best Hyper Parameters:\n", model1.best_params_)
    # Prediction
    prediction = model1.predict(X_test)
    # importing the metrics module
    from sklearn import metrics
    # evaluation(Accuracy)
    print("Accuracy:", metrics.accuracy_score(prediction, y_test))
    # evaluation(Confusion Metrix)
    cm = metrics.confusion_matrix(prediction, y_test, normalize='true')
    print("Confusion Metrix:\n", metrics.confusion_matrix(prediction, y_test))
    print(np.diagonal(cm).sum(), np.sum(cm), np.diagonal(cm).sum() / np.sum(cm))
    # conf_matr_show(cm)
    conf_matr_show(cm, title='KNN', fn='conf_matrix_knn_true', issave=True)


def model_pred():
    with open(f'{path}ml/xb2_model.pkl', 'rb') as f:
        model = pickle.load(f)


# df =


def format_cat(df):
    df['psfMagErr_u'] = df['psfMagErr_u'].round(5)
    df['psfMagErr_g'] = df['psfMagErr_g'].round(5)
    df['psfMagErr_r'] = df['psfMagErr_r'].round(5)
    df['psfMagErr_i'] = df['psfMagErr_i'].round(5)
    df['psfMagErr_z'] = df['psfMagErr_z'].round(5)
    df['psfMag_u'] = df['psfMag_u'].round(5)
    df['psfMag_g'] = df['psfMag_g'].round(5)
    df['psfMag_r'] = df['psfMag_r'].round(5)
    df['psfMag_i'] = df['psfMag_i'].round(5)
    df['psfMag_z'] = df['psfMag_z'].round(5)
    df['rowv'] = df['rowv'].round(5)
    df['rowvErr'] = df['rowvErr'].round(5)
    df['colv'] = df['colv'].round(5)
    df['colvErr'] = df['colvErr'].round(5)
    df['vel'] = df['vel'].round(5)
    df['velErr'] = df['velErr'].round(5)
    # df['ang_dist'] = df['ang_dist'].round(3)
    # df['dv_dec'] = df['dv_dec'].round(3)
    # df['dv_ra'] = df['dv_ra'].round(3)
    # df['dv_abs'] = df['dv_abs'].round(6)
    # df['mjd'] = df['mjd'].round(6)

    df['offsetRa_u'] = df['offsetRa_u'].round(5)
    df['offsetRa_g'] = df['offsetRa_g'].round(5)
    df['offsetRa_r'] = df['offsetRa_r'].round(5)
    df['offsetRa_i'] = df['offsetRa_i'].round(5)
    df['offsetRa_z'] = df['offsetRa_z'].round(5)
    df['offsetDec_u'] = df['offsetDec_u'].round(5)
    df['offsetDec_g'] = df['offsetDec_g'].round(5)
    df['offsetDec_r'] = df['offsetDec_r'].round(5)
    df['offsetDec_i'] = df['offsetDec_i'].round(5)
    df['offsetDec_z'] = df['offsetDec_z'].round(5)
    df['R2'] = df['R2'].round(2)

    df['ra_sb'] = df['ra_sb'].round(8)
    df['dec_sb'] = df['dec_sb'].round(8)
    df['ra'] = df['ra'].round(8)
    df['dec'] = df['dec'].round(8)

    df['ra_sb_rate'] = df['ra_sb_rate'].round(8)
    df['dec_sb_rate'] = df['dec_sb_rate'].round(8)
    df['x'] = df['x'].round(8)
    df['y'] = df['y'].round(8)
    df['z'] = df['z'].round(8)
    df['vx'] = df['vx'].round(8)
    df['vy'] = df['vy'].round(8)
    df['vz'] = df['vz'].round(8)

    df['posunc'] = df['posunc'].round(3)
    df['centerdist'] = df['centerdist'].round(3)
    df['geodist'] = df['geodist'].round(8)
    df['heliodist'] = df['heliodist'].round(8)

    for color in color_list:
        df[color] = df[color].round(5)

    df['Number'] = df['Number'].astype('Int64')
    df['type'] = df['type'].astype(np.int)

    return df


def my_predict(alg):
    sdss = pd.read_csv(f'{path}last/sso_tot4d.csv')

    # sdss = sdss[sdss['centerdist'] < 2.0]
    print('asteroids sdss:', len(set(sdss['Name'])))
    print('measurements sdss:', sdss.__len__())

    sdss.loc[(sdss['bcolor_ru']) & (sdss['bcolor_rg']), 'u-g'] = \
        sdss.loc[:, 'psfMag_u'] - sdss.loc[:, 'psfMag_g']
    sdss.loc[sdss['bcolor_ru'], 'u-r'] = \
        sdss.loc[:, 'psfMag_u'] - sdss.loc[:, 'psfMag_r']
    sdss.loc[(sdss['bcolor_ru']) & (sdss['bcolor_ri']), 'u-i'] = \
        sdss.loc[:, 'psfMag_u'] - sdss.loc[:, 'psfMag_i']
    sdss.loc[(sdss['bcolor_ru']) & (sdss['bcolor_rz']), 'u-z'] = \
        sdss.loc[:, 'psfMag_u'] - sdss.loc[:, 'psfMag_z']
    sdss.loc[(sdss['bcolor_rg']), 'g-r'] = \
        sdss.loc[:, 'psfMag_g'] - sdss.loc[:, 'psfMag_r']
    sdss.loc[(sdss['bcolor_rg']) & (sdss['bcolor_ri']), 'g-i'] = \
        sdss.loc[:, 'psfMag_g'] - sdss.loc[:, 'psfMag_i']
    sdss.loc[(sdss['bcolor_rg']) & (sdss['bcolor_rz']), 'g-z'] = \
        sdss.loc[:, 'psfMag_g'] - sdss.loc[:, 'psfMag_z']
    sdss.loc[(sdss['bcolor_ri']), 'r-i'] = \
        sdss.loc[:, 'psfMag_r'] - sdss.loc[:, 'psfMag_i']
    sdss.loc[(sdss['bcolor_rz']), 'r-z'] = \
        sdss.loc[:, 'psfMag_r'] - sdss.loc[:, 'psfMag_z']
    sdss.loc[(sdss['bcolor_ri']) & (sdss['bcolor_rz']), 'i-z'] = \
        sdss.loc[:, 'psfMag_i'] - sdss.loc[:, 'psfMag_z']

    sdss['pn'] = alg.predict(sdss[color_list])
    sdss['pc'] = sdss['pn'].replace(d_cls_inv)

    cond = sdss[color_list].isnull().all(axis=1)
    print(len(sdss[cond]))
    # sdss['predict'] = sdss['predict'][~cond] = 'U'
    print(sdss['pc'].value_counts(normalize=True))

    sdss = format_cat(sdss)


# sdss.to_csv(f'{path}last/sso_tot5a.csv', index=False)


def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=5, isshow=False):
    # if useTrainCV:
    #     xgb_param = alg.get_xgb_params()
    #     xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    #     cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
    #                       early_stopping_rounds=early_stopping_rounds, verbose_eval=True,
    #                       metrics='auc',
    #                       )
    #     alg.set_params(n_estimators=cvresult.shape[0])
    if useTrainCV:
        params = alg.get_xgb_params()
        xgb_param = dict([(key, [params[key]]) for key in params])

        boost = xgb.sklearn.XGBClassifier()
        cvresult = GridSearchCV(boost, xgb_param, cv=cv_folds)
        cvresult.fit(dtrain[predictors].values, dtrain[target].values)
        alg = cvresult.best_estimator_

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    mmc = metrics.matthews_corrcoef(dtrain[target].values, dtrain_predictions)
    # for item in  roauc:
    #     print(item)
    print("MMC Score (Train): %f" % mmc)
    cm = metrics.confusion_matrix(dtrain[target].values, dtrain_predictions, normalize='true')

    if isshow:
        feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        conf_matr_show(cm, title='XGBoost', fn='conf_matrix_true', issave=False)
        plt.show()

    return alg


def modelfit2(alg, X_train, y_train, X_test, y_test, useTrainCV=True, cv_folds=5, isshow=False):
    if useTrainCV:
        params = alg.get_xgb_params()
        xgb_param = dict([(key, [params[key]]) for key in params])

        boost = xgb.sklearn.XGBClassifier()
        cvresult = GridSearchCV(boost, xgb_param, cv=cv_folds)
        cvresult.fit(X_train, y_train)
        alg = cvresult.best_estimator_

    # Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(X_test)
    y_pred = alg.predict(X_train)
    # dtrain_predprob = alg.predict_proba(X_test)[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions))
    mmc = metrics.matthews_corrcoef(y_test, dtrain_predictions)
    # for item in  roauc:
    #     print(item)
    print("MMC Score (Test): %f" % mmc)
    cm = metrics.confusion_matrix(y_test, dtrain_predictions, normalize='true')
    cm_train = metrics.confusion_matrix(y_train, y_pred, normalize='true')
    mmc = metrics.matthews_corrcoef(y_train, y_pred)
    print("MMC Score (Train): %f" % mmc)

    print(metrics.classification_report(y_test, dtrain_predictions))

    if isshow:
        feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        # plt.savefig('./figs/future_importance.png', dpi=150)
        plt.show()
        conf_matr_show(cm, title='XGBoost', fn='conf_matrix_true', issave=False)
        conf_matr_show(cm_train, title='XGBoost', fn='conf_matrix_true', issave=False)
        plt.show()

    return alg


def tune_xboost1():
    combined = pd.read_csv(f'{path}ml/combined2.csv')
    # Choose all predictors except target & IDcols
    # predictors = color_list

    # y_init = combined[['complex']]
    # le = preprocessing.LabelEncoder()
    # combined['cl_number'] = le.fit_transform(y_init)
    # print(set(combined['cl_number']))
    # combined['cl_number'] = combined['complex']
    # combined['cl_number'] = combined['cl_number'].replace(d_cls)
    # target = 'cl_number'
    print(combined['complex'].value_counts(normalize=True, sort=False))
    X = combined[color_list]
    combined['cl_number'] = combined['complex']
    y = combined['cl_number'].replace(d_cls)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,
                                                        random_state=0)

    xgb1 = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=7,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        nthread=2,
        # scale_pos_weight=1,
        seed=0,
        num_class=10
    )
    model = modelfit2(xgb1, X, y, X_test, y_test, cv_folds=5, isshow=True)
    y_pred = model.predict(X)

    mmc = metrics.matthews_corrcoef(y, y_pred)
    print("MMC Score (Total): %f" % mmc)

    print(metrics.classification_report(y, y_pred))

    with open(f'{path}ml/xb_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    cm = metrics.confusion_matrix(y, y_pred, normalize='true')
    conf_matr_show(cm, title='XGBoost', fn='conf_matrix_xbs_model', issave=False)

    my_predict(model)


def tune_xboost2():
    combined = pd.read_csv(f'{path}ml/combined2.csv')
    predictors = color_list
    y_init = combined[['complex']]
    le = preprocessing.LabelEncoder()
    combined['cl_number'] = le.fit_transform(y_init)
    target = 'cl_number'

    param_test1 = {
        'max_depth': range(3, 10, 4),
        'min_child_weight': range(1, 6, 4)
    }

    xgb1 = xgb.XGBClassifier(learning_rate=0.1,
                             n_estimators=140,
                             max_depth=5,
                             min_child_weight=1,
                             gamma=0,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             objective='multi:softmax',
                             nthread=4,
                             scale_pos_weight=1,
                             seed=0)

    gsearch1 = GridSearchCV(estimator=xgb1,
                            param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(combined[predictors].values, combined[target].values)

    # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
    # Dictionary of best parameters
    best_pars = gsearch1.best_params_
    # Best XGB model that was found based on the metric score you specify
    best_model = gsearch1.best_estimator_
    print(best_pars)
    print(best_model)
    print(gsearch1.best_score_)


def gridtest1():
    # dataset = datasets.load_wine()
    # X = dataset.data
    # y = dataset.target
    dataset = pd.read_csv(f'{path}ml/combined2.csv')
    X = dataset[color_list]
    dataset['cl_number'] = dataset['complex']
    y = dataset['cl_number'].replace(d_cls)
    print(set(y))
    print(X.shape)
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,
                                                        random_state=0)

    # grid search
    # model = xgb.XGBClassifier()
    model = xgb.XGBClassifier(
        # learning_rate=0.1,
        # n_estimators=140,
        # max_depth=15,
        # min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        # nthread=4,
        # scale_pos_weight=1,
        # seed=0
    )
    n_estimators = [50, 100, 150, 200]
    learning_rate = [0.01, 0.05, 0.1, 0.2]
    max_depth = [3, 5, 7, 9]
    min_child_weight = [1, 3, 5, 7]

    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_child_weight=min_child_weight
                      )
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    grid_search = GridSearchCV(model, param_grid, scoring="f1_macro",
                               n_jobs=2, cv=kfold,
                               verbose=10)
    grid_result = grid_search.fit(X_train, y_train)

    # summarize results
    print()
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    # print(means)

    y_pred = grid_result.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred, normalize='true')
    conf_matr_show(cm, title='XGBoost', fn='conf_matrix_true', issave=False)

    print(metrics.classification_report(y_test, y_pred))
    acc = metrics.accuracy_score(y_test, y_pred)
    print('Predict accuracy: %f' % acc)

    from sklearn.metrics import classification_report
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    # plot results
    plt.subplots(figsize=(12, 12))
    # scores = np.array(means).reshape((len(learning_rate), len(n_estimators), len(max_depth)))
    #
    # for i, value in enumerate(learning_rate):
    #     plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    #
    # plt.legend()
    # plt.xlabel('n_estimators')
    # plt.ylabel('Log Loss')
    # plt.show()

    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.xlabel('colors')
    plt.show()


def training_features():
    combined = pd.read_csv(f'{path}ml/combined2.csv')
    colors = combined[color_list]
    a = colors.count().sort_values(ascending=False)
    print(a)
    a.plot(kind='bar', title='Number of Feature values')
    plt.savefig('./figs/numb_futures.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    path = f'/media/gamer/Data/data/sdss/init/'
    # d_cls = {"A": 0, "B": 1, "C": 2, "D": 3, "K": 4, "L": 5, "Q": 6, "S": 7, "U": 8, "V": 9, "X": 10}
    d_cls = {"A": 0, "B": 1, "C": 2, "D": 3, "K": 4, "L": 5, "Q": 6, "S": 7, "V": 8, "X": 9}
    d_cls_inv = {v: k for k, v in d_cls.items()}
    color_list = [
        'u-g', 'u-r', 'u-i', 'u-z',
        'g-r', 'g-i', 'g-z',
        'r-i', 'r-z',
        'i-z'
    ]
# print(d_cls.items())

# df = pd.read_csv(f'{path}last/sso_tot4c.csv')
# print('Load complete.')
# cond = df['bcolor_ru'] & df['bcolor_rg'] & df['bcolor_ri'] & df['bcolor_rz'] & \
#     df['bastrom_u'] & df['bastrom_g'] & df['bastrom_r'] & df['bastrom_i'] & df['bastrom_z']
# df = df[['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']][cond].reset_index()
# griz_show()
# color_combinations(df)
# coinc = load_class_from_spectra()
# taxonomy_show()
# knn_class()
# pca_analysis()
# feature_import()
# max_analysis()
# my_knn_analysis()
# max_me_phot()
# combining_knn()
# knn2()
# save_combined()
# xgboost_test()
# knn_hyper1()

# make_combined3()

# tune_xboost1()
# tune_xboost2()
# gridtest1()

# training_features()

