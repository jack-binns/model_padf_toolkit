import numpy as np
import matplotlib.pyplot as plt
import utils as u


class AtomicModel:

    def __init__(self, root='', tag='', sc_tag='', nn_dists=[0]):
        self.root = root
        self.tag = tag
        self.sc_tag = sc_tag
        self.nn_dists = nn_dists
        self.coord_nums = []
        self.bond_dists = []
        self.sp2_bond_dists = []
        self.sp3_bond_dists = []

        print(f'Looking at {self.tag}')
        self.subject_set = u.read_xyz(self.root + self.tag + '.xyz')
        self.super_set = u.read_xyz(self.root + self.sc_tag + '.xyz')

    def bonding_analysis(self, nbins=50):
        print(f'<bonding_analysis> Examining bonding with {self.nn_dists}')
        for probe in self.nn_dists:
            for atom in self.subject_set:
                nns = u.make_interaction_sphere(probe=probe, center=atom, atoms=self.super_set)
                self.coord_nums.append(len(nns))
                self.find_nn_dists(atom, nns)

            self.coord_nums = np.array(self.coord_nums)
            self.bond_dists = np.array(self.bond_dists)
            plt.figure()
            plt.title('Coordination numbers')
            plt.xlabel(f'# nearest neighbours up to {probe}')
            plt.ylabel('Counts')
            plt.hist(self.coord_nums, bins='auto')
            plt.figure()
            plt.title('Nearest neighbour bond distances')
            plt.xlabel(f'Bond Distance $\AA$')
            plt.ylabel('Counts')
            plt.hist(self.bond_dists,     bins=nbins, label='Total', edgecolor='None', alpha=0.5, color='g')
            plt.hist(self.sp3_bond_dists, bins=nbins, label='sp3', edgecolor='None', alpha=0.5, color='b')
            plt.hist(self.sp2_bond_dists, bins=nbins, label='sp2', edgecolor='None', alpha=0.5, color='r')
            plt.legend()
        plt.show()

    def find_nn_dists(self, center, atoms):
        for nn in atoms:
            dist = u.fast_vec_difmag(nn[0], nn[1], nn[2], center[0], center[1], center[2])
            self.bond_dists.append(dist)
            if len(atoms) == 4:
                self.sp3_bond_dists.append(dist)
            if len(atoms) == 3:
                self.sp2_bond_dists.append(dist)


if __name__ == '__main__':
    print('Examining bonding...')

    root = 'E:\\RMIT\\tom_harris\\'
    tag = '60xyz_final'         # original .xyz
    sc = '60xyz_sc-12-12-12'    # supercell

    model = AtomicModel(root=root, tag=tag, sc_tag=sc)
    model.nn_dists = [2.1]      #can add an extra distance if you want
    model.bonding_analysis()
