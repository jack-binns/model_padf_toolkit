import glob, os
import numpy as np
import math as m
from sklearn.cluster import KMeans

import utils


class XYZFile:

    def __init__(self, root_dir='foo', input_file='foo', out_tag='foo', sub_dict={}):
        self.root_dir = root_dir
        self.input_file = input_file
        self.raw_file = []
        self.out_tag = out_tag
        self.atom_sub_dict = sub_dict
        self.output_file = self.generate_output_string()
        self.atom_num = 0
        self.internal_comments = []
        self.atom_list = []

        self.frac_list = []
        self.cartesian_list = []
        self.atom_id_list = []
        self.atom_z_list = []

        self.mm = np.zeros((3, 3))
        self.invmm = np.zeros((3, 3))

        # Supercell params
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.super_a_min = 0
        self.super_a_max = 1
        self.super_b_min = 0
        self.super_b_max = 1
        self.super_c_min = 0
        self.super_c_max = 2
        self.supercell_atom_id_list = []
        self.supercell_atom_cart_list = []
        self.supercell_frac_list = []

        # self.grok_raw_file()

    def generate_output_string(self, folder=None):
        input_split = self.input_file.split('.')
        if not folder:
            self.output_file = f'{input_split[0]}{self.out_tag}.xyz'
        return self.output_file

    def multiframe_output_string(self, folder=None):
        input_split = self.input_file.split('.')
        # print(f'{input_split=}')
        input_path = input_split[0].split('\\')
        self.output_file = f'{self.input_file.split(".")[0]}\\{input_path[-1]}{self.out_tag}.xyz'

    def grok_raw_file(self):
        raw_x = []
        raw_y = []
        raw_z = []
        with open(self.input_file, 'r') as foo:
            self.raw_file = foo.readlines()
        # print(self.raw_file)
        self.atom_num = int(self.raw_file[0])
        self.internal_comments = self.raw_file[1]
        print(f'<grok_raw_file> {self.atom_num=}')
        for line in self.raw_file:
            ls = line.split()
            if len(ls) == 4:
                # it's an atom
                self.atom_id_list.append(ls[0])
                self.atom_z_list.append(utils.get_z(ls[0]))
                raw_x.append(ls[1])
                raw_y.append(ls[2])
                raw_z.append(ls[3])
                self.atom_list.append(ls)
        raw_x = [float(x) for x in raw_x]
        raw_y = [float(y) for y in raw_y]
        raw_z = [float(z) for z in raw_z]
        self.cartesian_list = np.column_stack((raw_x, raw_y, raw_z))

    def split_multiframe_xyz(self):
        print(f'splitting multi-frame xyz file')
        if not os.path.isdir(f'{self.input_file.split(".")[0]}\\'):
            os.mkdir(f'{self.input_file.split(".")[0]}\\')
        with open(self.input_file, 'r') as foo:
            self.raw_file = foo.readlines()
        self.atom_num = int(self.raw_file[0])
        frame_length = self.atom_num + 2
        ms_filelen = len(self.raw_file)
        print(ms_filelen)
        print(f'total of {ms_filelen / frame_length} {ms_filelen // frame_length} xyz files in ms')
        for n in range(ms_filelen // frame_length):
            frame_atoms = []
            print(f'creating file {n}')
            print(f'indices {n * (self.atom_num + 2)} {(n * self.atom_num + 2) + (self.atom_num + 2)}')
            raw_single_stage = self.raw_file[n * (self.atom_num + 2):n * (self.atom_num + 2) + (self.atom_num + 2)]
            print(f'{len(raw_single_stage)=}')
            print(f'{raw_single_stage[0]=}')
            print(f'{raw_single_stage[-1]=}')
            internal_comments = raw_single_stage[1]
            print(f'<grok_raw_file> {self.atom_num=}')
            for line in raw_single_stage:
                ls = line.split()
                if len(ls) == 4:
                    # it's an atom
                    ls[0] = ls[0][0]
                    frame_atoms.append(ls)
            self.out_tag = f'_frame_{n}'
            self.multiframe_output_string(folder='\\xyz_split\\')
            print(f'{self.output_file=}')
            with open(self.output_file, 'w') as foo:
                foo.write(f'{self.atom_num}\n')
                foo.write(f'{internal_comments}')
                for atom in frame_atoms:
                    atom_out = ''
                    for frag in atom:
                        atom_out += f'{frag} '
                    atom_out += '\n'
                    foo.write(atom_out)

    def clean_and_substitute(self):
        clean_list = []
        clean_id_list = []
        for k, atom in enumerate(self.atom_list):
            atom_type = atom[0]
            atom[0] = self.atom_sub_dict[atom_type]
            print(atom)
            clean_list.append(atom)
            clean_id_list.append(atom[0])
        self.atom_list = clean_list
        self.atom_id_list = clean_id_list
        print(f'<clean_and_substitute> {len(self.atom_list)} == {self.atom_num}')
        print(f'<clean_and_substitute> {len(self.atom_id_list)} == {self.atom_num}')
        # print(self.atom_id_list)

    def write_xyz_out(self, super_cell=False):
        if not super_cell:
            with open(self.output_file, 'w') as foo:
                foo.write(f'{self.atom_num}\n')
                foo.write(f'{self.internal_comments}')
                for atom in self.atom_list:
                    atom_out = ''
                    for frag in atom:
                        atom_out += f'{frag} '
                    atom_out += '\n'
                    foo.write(atom_out)
        if super_cell:
            print(f"{self.output_file=}")
            with open(self.output_file, 'w') as foo:
                foo.write(f'{len(self.supercell_atom_id_list)}\n')
                foo.write(
                    f'sc {self.super_a_min} {self.super_a_max} {self.super_b_min} {self.super_b_max} {self.super_c_min} {self.super_c_max}\n')
                for k, atom in enumerate(self.supercell_atom_cart_list):
                    atom_out = f'{self.supercell_atom_id_list[k]} '
                    for frag in atom:
                        atom_out += f'{frag:11.6f} '
                    atom_out += '\n'
                    foo.write(atom_out)

    def write_kmeansxyz_out(self, pixels, scaled_labels, outpath):
        print(f"<xyz_handler.write_kmeansxyz_out> {outpath=}")
        with open(outpath, 'w') as foo:
            foo.write(f'{len(pixels)}\n')
            foo.write('kmeans_scaled \n')
            for k, atom in enumerate(pixels):
                atom_out = f'{utils.get_id(scaled_labels[k])} '
                for frag in atom:
                    atom_out += f'{frag:11.6f} '
                atom_out += '\n'
                foo.write(atom_out)

    def generate_supercell(self):
        # create fractional coords for cell atoms
        print('<generate_supercell> generating supercell..')
        self.alpha = m.radians(self.alpha)
        self.beta = m.radians(self.beta)
        self.gamma = m.radians(self.gamma)
        n2 = (m.cos(self.alpha) - m.cos(self.gamma) * m.cos(self.beta)) / m.sin(self.gamma)
        self.mm = np.array([[self.a, self.b * m.cos(self.gamma), self.c * m.cos(self.beta)],
                            [0, self.b * m.sin(self.gamma), self.c * n2],
                            [0, 0, self.c * m.sqrt(m.sin(self.beta) ** 2 - n2 ** 2)]])
        self.invmm = np.linalg.inv(self.mm)
        print(self.mm)
        print(self.invmm)
        for k, cart_atom in enumerate(self.cartesian_list):
            frac_atom = self.invmm @ cart_atom
            self.frac_list.append(frac_atom)
            self.supercell_frac_list.append(frac_atom)
            self.supercell_atom_id_list.append(self.atom_id_list[k])
            # print(f"{self.atom_id_list[k]} {frac_atom}")

        # generate fractional coords for supercell atoms
        for a_ind in range(self.super_a_min, self.super_a_max):
            for b_ind in range(self.super_b_min, self.super_b_max):
                for c_ind in range(self.super_c_min, self.super_c_max):
                    if a_ind == 0 and b_ind == 0 and c_ind == 0:
                        # print('At cell')
                        continue
                    print(f' indices :{a_ind} , {b_ind}, {c_ind}')
                    for i, atom in enumerate(self.frac_list):
                        sc_x = atom[0] + a_ind
                        sc_y = atom[1] + b_ind
                        sc_z = atom[2] + c_ind
                        self.supercell_frac_list.append([sc_x, sc_y, sc_z])
                        self.supercell_atom_id_list.append(self.atom_id_list[i])
                        # print(f"{self.atom_id_list[i]} {[sc_x, sc_y, sc_z]}")

        print(f"generated the following super cell fractional coordinates...")
        print(f"{len(self.supercell_atom_id_list)=}")
        print(f"{len(self.supercell_frac_list)=}")

        # transfer frac coords to cartesian coords for all atoms
        for k, frac_atom in enumerate(self.supercell_frac_list):
            cart_sc_atom = self.mm @ np.array(frac_atom)
            self.supercell_atom_cart_list.append(cart_sc_atom)
            # print(f"{self.supercell_atom_id_list[k]} {cart_sc_atom}")
        self.write_xyz_out(super_cell=True)

    def gen_super(self):
        # First transfer all the cell atoms into the super cell list
        supercell_string_list = []
        for i, atom in enumerate(self.cartesian_list):
            supercell_string_list.append([self.atom_id_list[i], self.cartesian_list[i][0],
                                          self.cartesian_list[i][1], self.cartesian_list[i][2]])
            self.supercell_atom_cart_list.append([self.cartesian_list[i][0], self.cartesian_list[i][1],
                                                  self.cartesian_list[i][2]])
            self.supercell_atom_id_list.append(self.atom_id_list[i])
        print(len(self.supercell_atom_cart_list))
        print(len(self.supercell_atom_id_list))
        # Now generate the super cell
        for a_ind in range(self.super_a_min, self.super_a_max):
            for b_ind in range(self.super_b_min, self.super_b_max):
                for c_ind in range(self.super_c_min, self.super_c_max):
                    if a_ind == 0 and b_ind == 0 and c_ind == 0:
                        print('At cell')
                        continue
                    print(f' indices :{a_ind} , {b_ind}, {c_ind}')
                    for i, atom in enumerate(self.cartesian_list):
                        sc_x = atom[0] + (a_ind * self.a)
                        sc_y = atom[1] + (b_ind * self.b)
                        sc_z = atom[2] + (c_ind * self.c)
                        supercell_string_list.append([self.atom_id_list[i], sc_x, sc_y, sc_z])
        # print(supercell_string_list)
        self.atom_num = len(supercell_string_list)
        print(f'{len(supercell_string_list)=}')
        sc_set = set(tuple(x) for x in supercell_string_list)
        supercell_string_list = [list(x) for x in sc_set]
        print(f'{len(supercell_string_list)=}')
        base_split = self.output_file.split('.')
        self.atom_list = supercell_string_list
        self.output_file = f'{base_split[0]}_sc{self.atom_num}.xyz'
        self.write_xyz_out()

    def kmeans_fit(self, n_clusters=0):
        print("<xyz_handler.kmeans_fit> k-means clustering in progress")
        print(f"<xyz_handler.kmeans_fit> searching for {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.cartesian_list)
        # y_kmeans = kmeans.predict(self.cartesian_atoms)
        kmeans_pixels = kmeans.cluster_centers_
        kmeans_labels = kmeans.labels_
        # print(f'<xyz_handler.kmeans_fit> {kmeans_pixels=}')
        print(f'<xyz_handler.kmeans_fit> {len(kmeans_pixels)=}')
        # print(f'<xyz_handler.kmeans_fit> {kmeans_labels=}')
        print(f'<xyz_handler.kmeans_fit> {len(kmeans_labels)=}')
        return kmeans_pixels, kmeans_labels

    def label_kmeans_structure(self, pixels, labels, zmax=50):
        kmeans_z_list = []
        scaled_kmeans_z_list = []
        for k, pixel in enumerate(pixels):
            z_total = 0.0
            for n, label in enumerate(labels):
                if label == k:
                    z_total += utils.get_z(self.atom_id_list[n])
            kmeans_z_list.append(z_total)
        kmeans_z_list = np.array(kmeans_z_list)
        max_k_z = np.max(kmeans_z_list)
        # 50 is the highest z in the atomic_z dict
        [scaled_kmeans_z_list.append(int((kmeans_z_list[k] / max_k_z) * zmax)) for k, pixel in enumerate(pixels)]
        # [print(f'{k} {pixel} {int((kmeans_z_list[k] / max_k_z) * zmax)}') for k, pixel in enumerate(pixels)]
        return scaled_kmeans_z_list

    def pair_dist_calculation(self, atom_list=None):
        print(f'<xyz_handler.pair_dist_calculation> Calculating pairwise interatomic distances...')
        # interatomic_vectors
        n2_contacts = []
        interatomic_vectors = []
        for j, a_i in enumerate(atom_list):
            for a_j in atom_list:
                if not np.array_equal(a_i, a_j):
                    # print(a_i, a_j)
                    mag_r_ij = utils.fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                    # r_ij = utils.fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                    # r_ij.append(mag_r_ij)
                    # r_ij.append(a_i[3] * a_j[3])
                    n2_contacts.append(mag_r_ij)
                    # interatomic_vectors.append(r_ij)
        np.array(n2_contacts)
        print(f'<xyz_handler.pair_dist_calculation> {len(n2_contacts)} interatomic vectors')
        return n2_contacts


if __name__ == '__main__':
    root = 'C:\\rmit\\dlc\\model_padf\\md\\'
    xyz_file = root + '192DLC.xyz'

    multi_state_xyz = XYZFile(root_dir=root, input_file=xyz_file, out_tag='_conv')

    multi_state_xyz.split_multiframe_xyz()
    dlc_a = 96.872
    dlc_b = 42.096
    dlc_c = 117.520
    dlc_alpha = 90.00
    dlc_beta = 107.08
    dlc_gamma = 90.00

    # sub_dict = {'1': 'Al',
    #             '2': 'O'}

    file_list = glob.glob(f"{root}\\192DLC\\*.xyz")
    print(file_list)

    for k, xyz_file in enumerate(file_list[:]):
        xyz = XYZFile(root_dir=f'{root}\\192DLC\\', input_file=xyz_file, out_tag=f'_sc')
        xyz.a = dlc_a
        xyz.b = dlc_b
        xyz.c = dlc_c
        xyz.alpha = dlc_alpha
        xyz.beta = dlc_beta
        xyz.gamma = dlc_gamma
        xyz.super_a_min = 0
        xyz.super_a_max = 1
        xyz.super_b_min = 0
        xyz.super_b_max = 2
        xyz.super_c_min = 0
        xyz.super_c_max = 1
        xyz.grok_raw_file()
        xyz.generate_supercell()

    # root = 'C:\\rmit\\alox\\'
    # file_list = glob.glob(root + '*.xyz')
    #
    # sub_dict = {'1': 'Al',
    #             '2': 'O'}
    #
    # for xyz_file in file_list:
    #     xyz = XYZFile(root_dir=root, input_file=xyz_file, out_tag='_conv', sub_dict=sub_dict)
    #     xyz.super_a_min = -1
    #     xyz.super_a_max = 2
    #     xyz.super_b_min = -1
    #     xyz.super_b_max = 2
    #     xyz.super_c_min = 0
    #     xyz.super_c_max = 1
    #     xyz.a = 24
    #     xyz.b = 24
    #     xyz.c = 48
    #
    #     xyz.clean_and_substitute()
    #     xyz.gen_super()
    #     xyz.write_xyz_out()
