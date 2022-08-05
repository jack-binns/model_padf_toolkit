import xyz_handler
import utils
import glob

if __name__ == '__main__':
    """
    -------------------------------------------------
    .xtc trajectory model PADF calculation setup tool
    -------------------------------------------------
    
    Starting point: stacked .xyz file containing each frame of the trajectory.
    
    1. Splits the stacked .xyz file into n separate .xyz files - this only needs to be done once. Also rejects H atoms.
    2. Generates a supercell according to the [super_d_min/max] values
    3. Performs a k-means clustering fit to distribute [n_clusters] pseudo atoms within the super cell. The atomic
        weight of the substituted atoms is taken into account and replaced by a pseudo-atom with weighted z.
    4. This process is repeated for each file in a selected subset, which is ready for processing by the model_padf
        controller unit which manages the model PADF calculation for the trajectory
    """

    """
    Splitting trajectory .xyz file exported by VMD
    """
    tag = '192DLC_50ns'
    root_dir = f'C:\\rmit\\dlc\\model_padf\\md\\{tag}\\'
    trajectory_xyz_filename = f'{root_dir}{tag}.xyz'

    trajectory_xyz = xyz_handler.XYZFile(root_dir=root_dir, input_file=trajectory_xyz_filename,
                                         out_tag='_conv',
                                         drop_h=True)

    trajectory_xyz.split_multiframe_xyz()

    """
    Collect the single-frame xyz files for the next steps
    """
    single_frame_list = utils.sorted_nicely(glob.glob(f"{root_dir}\\{tag}\\*.xyz"))
    print(f'First frame: {single_frame_list[0]}')
    print(f'Last frame: {single_frame_list[-1]}')

    """
    Supercell generation & clustering
    """

    n_clusters = 300  # The number of pseudo-atoms that will be placed into the supercell to approximate the
    # underlying electron density

    # Set of unit cell dimensions, can be non-orthonormal
    ucd_a = 96.872
    ucd_b = 42.096
    ucd_c = 117.520
    ucd_alpha = 90.00
    ucd_beta = 107.08
    ucd_gamma = 90.00
    # Number of additional cell lengths to generate supercell.
    # An **unexpanded** cell is defined as ((0,1), (0,1), (0,1))
    super_a_min = 0
    super_a_max = 1
    super_b_min = 0
    super_b_max = 2  # Here in this example we are expanding along the b axis.
    super_c_min = 0
    super_c_max = 1

    # Here we do the processing
    for k, xyz_file in enumerate(single_frame_list[:]):
        xyz = xyz_handler.XYZFile(root_dir=f'{root_dir}\\{tag}\\', input_file=xyz_file, out_tag=f'_sc')
        xyz.a = ucd_a
        xyz.b = ucd_b
        xyz.c = ucd_c
        xyz.alpha = ucd_alpha
        xyz.beta = ucd_beta
        xyz.gamma = ucd_gamma
        xyz.super_a_min = super_a_min
        xyz.super_a_max = super_a_max
        xyz.super_b_min = super_b_min
        xyz.super_b_max = super_b_max
        xyz.super_c_min = super_c_min
        xyz.super_c_max = super_c_max
        xyz.grok_raw_file()
        xyz.generate_supercell()
        pix, labels = xyz.kmeans_fit(n_clusters=n_clusters, cartesian_list=xyz.supercell_atom_cart_list)
        zs_labels = xyz.label_kmeans_structure(pixels=pix, labels=labels, atom_id_list=xyz.supercell_atom_id_list)
        xyz.write_kmeansxyz_out(pixels=pix, scaled_labels=zs_labels,
                                outpath=f"{root_dir}\\{tag}\\{tag}_frame_{k}_sc_{n_clusters}.xyz")
