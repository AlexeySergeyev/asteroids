import pandas as pd
import matplotlib.pyplot as plt

def plot_gr_iz(df):
    _ = plt.figure(figsize=(10, 9))
    complex_set = set(df['complex'])
    complex = list(complex_set)
    complex.sort()
    for i, letter in enumerate(complex[:]):
        cond = df['complex'] == letter
        temp = ref[cond].reset_index()
        plt.scatter(temp['psfMag_r'] - temp['psfMag_i'],
                    temp['psfMag_i'] - temp['psfMag_z'],
                    label=letter, marker=markers[i], s=50
                    )
    plt.xlabel('g-r')
    plt.ylabel('i-z')
    # plt.xlim([0.3, 0.9])
    # plt.ylim([-0.6, 0.6])
    plt.legend(loc='lower right')
    # plt.savefig(name)
    plt.show()

if __name__ == '__main__':
    markers = ['.', '+', 'x', 'v', '^', '1', '2', '3', '4', 's', '*']

    ref = pd.read_csv('taxo_ref.csv')
    plot_gr_iz(ref)