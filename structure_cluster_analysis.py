import numpy as np

import xyz_handler as xh
import matplotlib.pyplot as plt

plt.rcParams['axes.linewidth'] = 0.2  # set the value globally
plt.rcParams["font.family"] = "Arial"


class ClusterAnalyzer:

    def __init__(self, root_dir='foo', input_file='foo'):
        self.root_dir = root_dir
        self.input_file = input_file

        self.xyz = xh.XYZFile(root_dir=self.root_dir, input_file=self.input_file)
        self.xyz.grok_raw_file()

    def iav_count_analyzer(self, nmin, nmax, niter, rmax=None):
        n_cluster_space = np.linspace(start=nmin, stop=nmax, num=niter)
        r_space = np.linspace(start=0, stop=rmax, num=25)
        ensemble_niavs = []
        ensemble_niav_hists = []
        ensemble_niavs_mags = np.zeros(len(n_cluster_space))

        plt.figure()
        plt.xlabel('r (Å or nm)')
        plt.ylabel('Number of intercluster vectors')
        for j, n_clusters in enumerate(n_cluster_space):
            n_clusters = int(n_clusters)
            pix, labels = ca.xyz.kmeans_fit(n_clusters=n_clusters)
            zs_labels = ca.xyz.label_kmeans_structure(pixels=pix, labels=labels)
            ca.xyz.write_kmeansxyz_out(pixels=pix, scaled_labels=zs_labels,
                                       outpath=f'{root}192DLC_kmeans_{n_clusters}.xyz')
            pds = ca.xyz.pair_dist_calculation(atom_list=pix)
            ensemble_niavs.append(pds)
            ensemble_niavs_mags[j] = len(pds)
            pds = np.array(pds)
            niav_hist = np.histogram(pds, range=(0, rmax), bins=25)
            plt.plot(r_space, niav_hist[0], label=f'{n_clusters} clusters')
        plt.legend()
        ensemble_niavs_mags = np.array(ensemble_niavs_mags)
        print(f'{ensemble_niavs_mags.shape=}')
        plt.figure()
        plt.ylabel('Number of intercluster vectors')
        plt.xlabel('Number of clusters')
        plt.plot(n_cluster_space, ensemble_niavs_mags)
        plt.show()


if __name__ == '__main__':
    print('XYZ kMeans cluster analysis')

    root = 'C:\\rmit\\dlc\\model_padf\\'
    file = f'{root}192DLC_frame_1_sc.xyz'
    clusters = 10000

    ca = ClusterAnalyzer(root_dir=root, input_file=file)
    ca.iav_count_analyzer(nmin=100, nmax=10000, niter=10, rmax=40.0)

    # pix, labels = ca.xyz.kmeans_fit(n_clusters=clusters)
    # zs_labels = ca.xyz.label_kmeans_structure(pixels=pix, labels=labels)
    # ca.xyz.write_kmeansxyz_out(pixels=pix, scaled_labels=zs_labels, outpath=f'{root}192DLC_kmeans_{clusters}.xyz')
