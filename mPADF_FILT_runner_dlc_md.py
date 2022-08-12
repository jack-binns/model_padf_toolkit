import mPADF_plot as mpp
# import mPADF_plot_unstable as mpp
import matplotlib.pyplot as plt

plt.rcParams['axes.linewidth'] = 0.5  # set the value globally
plt.rcParams["font.family"] = "Arial"

if __name__ == '__main__':
    print("Plotting model PADF slices...")
    # root = 'E:\\RMIT\\mofs_saw\\ZIF8_mPADF\\'
    # root = 'E:\\RMIT\\mofs_saw\\HKUST1_mPADF\\'
    # root = 'E:\\RMIT\\mofs_saw\\MIL53_mPADF\\'
    # root = 'E:\\RMIT\\dlc\\model_padf\\'
    root = 'C:\\rmit\\dlc\\model_padf\\md\\192DLC_50ns\\'
    # project = "defect1\\"
    project = "600cluster_contar0p6_nframes250\\"
    # tag = "ZIF8_ordered"
    tag = "192DLC_50ns_trajectory"

    ### Details about the filters
    filter_config_path = root + project + f'{tag}_filter_config.txt'
    filter_lmax = 30  # can be as big as nl in the filter config

    ### Generating the model PADF handler
    model_plotter = mpp.ModelPADF(root=root, project=project, tag=tag)
    model_plotter.r_plim = 40.0  # plotting limit in r

    # Perform filter calculation
    # model_plotter.filter_mPADF(lmax=filter_lmax)

    # model_plotter.show_reqr(filter=True, show=False, char_dists=[], radial_scaling=0,aspect=3)
    model_plotter.show_reqr(filter=False, show=False, char_dists=[], radial_scaling=0, aspect=3)

    # nff_reqr = model_plotter.finite_show_reqr(filter=False, radial_scaling=0, r_tol=2.5, char_dists=[], aspect=3)
    # ff_reqr = model_plotter.finite_show_reqr(filter=True, radial_scaling=0, r_tol=2.5, char_dists=[], aspect=3)
    #
    # r_of_interest = [5.9, 8, 12]
    # model_plotter.line_profile_plot(target_r=r_of_interest, r_tol=0.0, filter=False, dump=True)
    # model_plotter.line_profile_plot(target_r=r_of_interest, r_tol=0.0, filter=True, dump=True)
    # model_plotter.line_profile_plot(target_r=r_of_interest, r_tol=0.0, arr=nff_reqr, array_tag='nofilt_finite',
    #                                 dump=True)
    # model_plotter.line_profile_plot(target_r=r_of_interest, r_tol=0.0, arr=ff_reqr, array_tag='filt_finite', dump=True)

    plt.show()
