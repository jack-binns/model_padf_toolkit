import matplotlib.pyplot as plt
import numpy as np
import paramsMPADF
import paramsFILT
import generate_sphBzeros_mPADF
import padflibdev as pld
import os
import glob
import scipy.special as sps
from scipy.ndimage.filters import gaussian_filter as sp_gaussian_filter
from scipy import signal


def gamma_scale(disp, gamma):
    disp = disp / gamma
    return disp


def project_padf_Legendre(padf_vol, l_ind):
    s2 = padf_vol.shape[2]
    z = np.cos(2.0 * np.pi * np.arange(s2) / float(s2))
    Pn = sps.legendre(int(l_ind))
    p = Pn(z)
    print(f"<project_padf_Legendre> pshape {p.shape} {padf_vol.shape}")
    mat = np.zeros(padf_vol.shape)
    for i in np.arange(padf_vol.shape[0]):
        for j in np.arange(padf_vol.shape[1]):
            mat[i, j, :] = p[:]
    output = np.sum(mat * padf_vol, 2)
    return output


def calcBlrr(padf_vol, lmax):
    blrr = []
    for l_ind in np.arange(lmax):
        blrr.append(project_padf_Legendre(padf_vol, l_ind))
    return blrr


def filter_Blrr(blrr_vol, filters):
    for l_ind in np.arange(len(blrr_vol)):
        blrr_vol[l_ind] = np.dot(np.dot(filters[l_ind], blrr_vol[l_ind]), filters[l_ind].transpose())
    return blrr_vol


def Blrr_to_padf(blrr_vol, padfshape):
    padfout = np.zeros(padfshape)
    lmax = len(blrr_vol)
    for l_ind in np.arange(2, lmax):
        if (l_ind % 2) != 0:
            continue
        s2 = padfshape[2]
        z = np.cos(2.0 * np.pi * np.arange(s2) / float(s2))
        Pn = sps.legendre(int(l_ind))
        p = Pn(z)
        for i in np.arange(padfshape[0]):
            for j in np.arange(padfshape[1]):
                padfout[i, j, :] += blrr_vol[l_ind][i, j] * p[:]
    return padfout


class ModelPADF:

    def __init__(self, root='foo', project='/', tag='tag', section=1, theta=1.0, r_plim=1.0, r2=0.0,
                 filter_config='',
                 theta_mask=0.0):
        self.root = root
        self.project = project
        self.tag = tag
        self.title = ''

        self.params_model = self.read_mpadf_params()
        self.rmax = self.params_model.rmax
        self.nr = self.params_model.nr
        self.nth = self.params_model.nth
        self.angular_bin = 180.0 / self.nth
        self.r_dist_bin = self.rmax / self.nr
        self.padf = np.zeros(0)
        self.disp_padf = np.zeros(0)
        self.disp_padf_reqr = np.zeros(0)
        self.exp_padf = np.zeros(0)  # expanded padf generated from corrections

        # padf plot parameters
        self.section = section
        self.theta = theta
        self.r_plim = r_plim  # r limit for plotting
        self.r2 = r2
        self.gnuplot = False
        self.theta_mask = theta_mask

        # filter params
        self.lmax = 0
        self.filter_config = filter_config
        self.filter_path = ''
        self.params_filt = ''
        self.filter_padf = np.zeros(0)
        self.filters = []
        self.filt_padf = np.zeros(0)
        self.disp_filt_padf = np.zeros(0)
        self.disp_filt_padf_reqr = np.zeros(0)

        # read in the mPADF and correct
        self.read_mpadf()
        self.geom_corrections()

    def sin_theta_correction_slice(self, padf_vol):
        print("<sin_theta_correction_slice> Applying sin(theta) correction...")
        factor_array = np.ones(padf_vol.shape)
        sinbins = 180.0 / (self.nth - 1)
        for th in range(self.nth):
            angle = th * sinbins
            if 0.0 < angle < 180.0:
                factor_array[:, :, th] = np.sin(np.deg2rad(angle))
            else:
                factor_array[:, :, th] = 1.0
        trans_vol = padf_vol / factor_array
        return trans_vol

    def geom_corrections(self):
        """
        Perform symmetrization and geometric corrections to atomistic mPADF
        """
        print('<geom_corrections> Applying geometric corrections...')

        padf = self.padf
        if self.theta_mask != 0.0:
            theta_index = int((self.theta_mask * self.nth) / 180.0)
            masked_padf = np.zeros(shape=padf.shape)
            print(f'<geom_corrections> applying theta mask {self.theta_mask} :: {theta_index}')
            for th in np.arange(self.nth):
                th_slice = padf[:, :, th]
                if theta_index < th < (self.nth - theta_index):
                    masked_padf[:, :, th] = th_slice
            padf = masked_padf
        flipped_padf = padf[:, :, ::-1]
        padf = padf + flipped_padf
        flipped_only_padf = np.copy(padf)
        padf = self.sin_theta_correction_slice(padf)
        ## Scaling
        dr = 0.1
        r = np.arange(self.nr) * dr
        r2 = np.zeros(padf.shape)
        for ith in np.arange(self.nth):
            r2[:, :, ith] = np.outer(r, r)
        ir = np.where(r2 > 0.0)
        padf[ir] *= 1.0 / (r2[ir] ** 1.0)
        # subtract theta average
        padf_meansub = np.copy(padf)
        for ir in np.arange(self.nr):
            for ir2 in np.arange(self.nr):
                padf_meansub[ir, ir2, :] += -np.average(padf_meansub[ir, ir2, :])
        ex_nth = 2 * self.nth
        padf_extend = np.zeros((self.nr, self.nr, ex_nth))
        padf_extend[:, :, :self.nth] = self.padf                    ##
        self.exp_padf = padf_extend
        # Disp PADF is always the atomistic PADF with the full set of geometry corrections
        self.disp_padf = padf_meansub
        exp_padf_reqr = np.zeros(shape=(self.nr, self.nth * 2))
        for i in range(self.nr):
            exp_padf_reqr[i, :] = self.exp_padf[i, i, :]

        disp_padf_reqr = np.zeros(shape=(self.nr, self.nth))
        for i in range(self.nr):
            disp_padf_reqr[i, :] = self.disp_padf[i, i, :]
        print(f'<geom_corrections> expanded padf.shape{self.exp_padf.shape} ')
        print(f'<geom_corrections> padf.shape{self.padf.shape} ')
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum_geomcorr.npy', self.disp_padf)
        print(
            f'<geom_corrections> geometry corrected mPADF volume saved to : {self.root + self.project + self.tag}_mPADF_total_sum_geomcorr.npy')

    def read_mpadf_params(self):
        print('<read_mpadf_params> Reading parameter file...')
        p = paramsMPADF.paramsMPADF()
        p.read_calc_param_file(f'{self.root + self.project + self.tag}_mPADF_param_log.txt')
        return p

    def read_filt_params(self):
        print('<read_filt_params> Reading filter parameter file...')
        p = paramsFILT.paramsFILT()
        p.read_config_file(self.filter_config)
        return p

    def read_filters(self, nr=0, lmax=0):
        for l_ind in np.arange(lmax):
            print(self.filter_path + self.tag + "_l" + str(l_ind) + "_filter.npy")
            filter = np.load(self.filter_path + self.tag + "_l" + str(l_ind) + "_filter.npy").reshape(nr, nr)
            self.filters.append(filter)

    def read_mpadf(self):
        print('<read_mpadf> Reading in PADF array...')
        self.padf = np.load(self.root + self.project + self.tag + '_mPADF_total_sum.npy')
        print(f'<read_mpadf> {self.padf.shape}...')
        print(f'<read_mpadf> {np.max(self.padf)}...')

    def radial_correction(self, arr, rpower):
        dr = 0.1
        r = np.arange(self.nr) * dr
        r2 = np.zeros(arr.shape)
        arr_d = np.copy(arr)
        for ith in np.arange(self.nth):
            r2[:, :, ith] = np.outer(r, r)
        ir = np.where(r2 > 0.0)
        arr_d[ir] *= 1.0 / (r2[ir] ** rpower)
        return arr_d

    def show_reqr(self, aspect=10, show=False, char_dists=None, filter=False,
                  radial_scaling=0, gaussian_blur=0,
                  clims=(0, 1), vpadf=None, title=''):
        """
        Takes a 3D padf volume and plots the r = r' slice. Can be overlaid with characteristic
        distance chords given in the list char_dists
        :param radial_scaling: r power used to scale padf intensity in r
        :param aspect: aspect ratio for plt.imshow
        :param show: boolean, immediately display the plot. Turn off to embed output into figure series
        :param char_dists: characteristic interscatterer distance to be plotted
        :return: Can return the slice if required for other purposes. ie. saving
        """
        print(f'self.nth {self.nth}')
        print(f'self.nr {self.nr}')
        vpadf_d = self.radial_correction(vpadf, radial_scaling)
        disp = np.zeros((self.nr, self.nth))
        print(f'disp.shape  {disp.shape}')
        print(f'vpadf.shape {vpadf_d.shape}')
        for i in np.arange(self.nr):
            disp[i, :] = vpadf_d[i, i, :]
        plt.figure()
        if len(char_dists) > 0:
            th_range = np.linspace(1, 180, 180)
            for rbc in char_dists:
                f_th = rbc / (2 * np.sin(np.radians(th_range / 2)))
                plt.plot(th_range, f_th, dashes=[6, 2], label=str(rbc))
                plt.legend()
                plt.ylim(0, self.r_plim)
        if title:
            plt.title(title)
        if gaussian_blur > 0:
            disp = sp_gaussian_filter(disp, sigma=gaussian_blur)
        plt.xlabel(r'${\Theta}$ / $^\circ$')
        plt.ylabel(r'$r = r^\prime$ / $\mathrm{\AA}$')
        plt.imshow(disp[:int((self.r_plim / self.rmax) * self.nr), :], extent=[0, 180, 0, self.r_plim], origin='lower',
                   aspect=aspect)
        plt.clim(clims[0], clims[1])
        if show:
            plt.show()
        if filter:
            self.disp_filt_padf_reqr = disp[:int((self.r_plim / self.rmax) * self.nr), :]
        else:
            self.disp_padf_reqr = disp[:int((self.r_plim / self.rmax) * self.nr), :]
        if self.gnuplot:
            gamma_d = gamma_scale(disp, np.max(disp))
            np.savetxt(self.root + self.tag + '_reqr_gp.txt', gamma_d)
            print(f"gnuplot output written to {self.root + self.tag + '_reqr_gp.txt'}")
        return disp[:int((self.r_plim / self.rmax) * self.nr), :]

    def filter_mPADF(self, lmax=30):
        # Set up the filter folder
        self.lmax = lmax
        print(f'<filter_mPADF> Setting up blfilters...')
        self.filter_path = self.root + self.project + 'blfilt\\'
        if not os.path.isdir(self.filter_path):
            os.mkdir(self.filter_path)
        print(f'<filter_mPADF> Filters stored in {self.filter_path}')
        # Read in filter parameters
        self.params_filt = self.read_filt_params()
        # Check if the sphBzeros already exist
        zero_list = glob.glob(self.filter_path + '*.npy')
        if len(zero_list) == 0:
            print('<filter_mPADF> Generating sphbzeroes...')
            generate_sphBzeros_mPADF.gen_sphBzeros(self.filter_path, lmax=self.params_filt.nl)
        # Set up an instance of the PADF class
        print(f"<filter_mPADF> filter qmax {self.params_filt.qmax}")
        print(f"<filter_mPADF> filter rmax {self.params_filt.rmax}")
        print(f"<filter_mPADF> filter tablepath {self.filter_path}")
        self.filter_padf = pld.padfcl(nl=self.params_filt.nl, nr=self.params_filt.nr, nq=self.params_filt.nq,
                                      qmin=self.params_filt.qmin, qmax=self.params_filt.qmax,
                                      rmax=self.params_filt.rmax, tablepath=self.filter_path,
                                      tablelmax=self.params_filt.nl)
        # Calculate the filter files
        print(f"<filter_mPADF> Calculating filters...")
        print(f"<filter_mPADF> - nl : {self.params_filt.nl}")
        print(f"<filter_mPADF> - rmax : {self.params_filt.rmax}")
        print(f"<filter_mPADF> - qmax : {self.params_filt.qmax}")
        print(f"<filter_mPADF> - qmin : {self.params_filt.qmin}")
        filter_list = glob.glob(self.filter_path + self.tag + "_l*_filter.npy")
        if len(filter_list) == 0:
            for l_ind in range(self.params_filt.nlmin, self.params_filt.nl, 1):
                nmax = self.filter_padf.blqq.sphB_samp_nmax(l_ind, self.params_filt.rmax, self.params_filt.qmax)
                print(f"<filter_mPADF> l_ind, nmax  :  {l_ind}, {nmax}")
                dsB = self.filter_padf.dsBmatrix(l_ind, nmax)
                dsBinv = self.filter_padf.dsBmatrix_inv(l_ind, nmax, self.params_filt.qmin)
                filtermat = np.dot(dsB, dsBinv)
                outname = self.filter_path + self.tag + "_l" + str(l_ind) + "_filter.npy"
                np.save(outname, filtermat)
                outname = self.filter_path + self.tag + "_l" + str(l_ind) + "_dsB.npy"
                np.save(outname, dsB)
            print(f"<filter_mPADF> ...filters written to  : {self.filter_path}{self.tag}_l*_filter.npy")
        # Main computation using filters
        self.read_filters(nr=self.params_filt.nr, lmax=lmax)
        # print(self.filters)
        print(f"<filter_mPADF> Calculating filtered mPADF...")
        blrr = calcBlrr(self.exp_padf, lmax)
        blrr_filt = filter_Blrr(blrr, self.filters)
        self.filt_padf = Blrr_to_padf(blrr_filt, self.exp_padf.shape)
        print(f'{self.filt_padf.shape=}')
        th_lim = self.filt_padf.shape[-1] // 2
        self.disp_filt_padf = self.filt_padf[:, :, :th_lim]
        print(f'{self.disp_filt_padf.shape=}')

    def line_profile_plot(self, target_r=None, r_tol=0.0, show=False, filter=True, dump=False, arr=[], array_tag=''):
        plt.figure()
        plt.xlabel(r'$\theta$ / $^\circ$')
        plt.ylabel(r'$\Theta(r = r^\prime)$')
        plt.xlim(0, 180)
        if len(arr) == 0:
            if filter:
                arr = self.disp_filt_padf_reqr
                of = 'FILT'
            else:
                arr = self.disp_padf_reqr
                of = ''
        else:
            arr = arr
            of = array_tag
        print(f'{arr.shape}  arr shape')
        for roi in target_r:
            pix_per_r = self.nr / self.rmax
            target_index = int(roi * pix_per_r)
            if r_tol > 0.0:
                target_min_index = int((roi - r_tol) * pix_per_r)
                target_max_index = int((roi + r_tol) * pix_per_r)
                slicer = np.sum(arr[target_min_index:target_max_index, :], axis=0)
                plt.plot(slicer, label=str(f'{roi}({r_tol})'))
                if dump:
                    np.savetxt(f'{self.root}{self.project}{self.tag}_lineplot_{of}_r{roi}.txt', slicer)

            else:
                slicer = arr[target_index, :]
                plt.plot(slicer, label=str(roi))
                if dump:
                    np.savetxt(f'{self.root}{self.project}{self.tag}_lineplot_{of}_r{roi}.txt', slicer)
        plt.legend()
        if show:
            plt.show()

    def add_diag_slice(self, idx, reqr, padf, verbose=False):
        if idx == 0:
            return reqr
        else:
            print(f'shape of the rolling slice {reqr.shape}')
            dslice = np.transpose(np.diagonal(padf, offset=idx, axis1=0, axis2=1)).copy()
            dslice += 2.0
            print(f'shape of the diag slice {dslice.shape}')
            if verbose:
                plt.figure()
                plt.title(f'before {idx}')
                plt.imshow(reqr)
                plt.figure()
                plt.title(f'dslice {idx}')
                plt.imshow(dslice)
            if idx % 2 == 0:
                new_slice = reqr.copy()
                print(new_slice[(idx // 2):(-idx // 2), :].shape)
                print(dslice.shape)
                new_slice[(idx // 2):(-idx // 2), :] += (dslice[:] * 2)
                if verbose:
                    plt.figure()
                    plt.title(f'after {idx}')
                    plt.imshow(new_slice)
            else:
                new_slice = reqr.copy()
                start_idx = int(np.floor(idx / 2))
                print(new_slice.shape)
                print(dslice.shape)
                for k, slindx in enumerate(np.arange(dslice.shape[0])):
                    new_slice[start_idx + slindx, :] += dslice[slindx, :]
                if verbose:
                    plt.figure()
                    plt.title(f'after {idx}')
                    plt.imshow(new_slice)
            return new_slice

    def finite_show_reqr(self, aspect=10, filter=False, radial_scaling=0,
                         r_tol=0.0, verbose=False, char_dists=None,
                         show=False, clims=(0, 1)):
        if filter:
            vpadf = self.disp_filt_padf
        else:
            vpadf = self.disp_padf
        vpadf = self.radial_correction(vpadf, rpower=radial_scaling)
        print(vpadf.shape)
        scr_reqr = np.zeros((self.nr, self.nth))

        ## Define scratch reqr slice
        for i in np.arange(self.nr):
            scr_reqr[i, :] = vpadf[i, i, :]

        pix_per_r = self.nr / self.rmax

        target_max_index = int(r_tol * pix_per_r)
        print(f'Extracting and summing slices up to {target_max_index}; ({r_tol} \AA * {pix_per_r} pix \AA^-1)')
        for target_index in np.arange(target_max_index + 1):
            scr_reqr = self.add_diag_slice(idx=target_index, reqr=scr_reqr, padf=vpadf, verbose=verbose)

        plt.figure()
        if len(char_dists) > 0:
            th_range = np.linspace(1, 180, 180)
            for rbc in char_dists:
                f_th = rbc / (2 * np.sin(np.radians(th_range / 2)))
                # f_th_r = rbc / (2 * np.cos(np.radians(th_range / 2)))
                plt.plot(th_range, f_th, dashes=[6, 2], label=str(rbc))
                plt.legend()
                # plt.plot(th_range, f_th_r)
                plt.ylim(0, self.r_plim)
        if self.title:
            plt.title(self.title)
        plt.xlabel(r'${\Theta}$ / $^\circ$')
        plt.ylabel(r'$r = r^\prime$ / $\mathrm{\AA}$')
        plt.imshow(scr_reqr[:int((self.r_plim / self.rmax) * self.nr), :], extent=[0, 180, 0, self.r_plim],
                   origin='lower',
                   aspect=aspect)
        plt.clim(clims[0], clims[1])
        if show:
            plt.show()
        if self.gnuplot:
            gamma_d = gamma_scale(scr_reqr, np.max(scr_reqr))
            np.savetxt(self.root + self.tag + '_reqr_gp.txt', gamma_d)
            print(f"gnuplot output written to {self.root + self.tag + '_reqr_gp.txt'}")
        return scr_reqr

    def theta_slices(self, theta_vals=[], arr='', clims=(1, 1), cmap='viridis', show=False, gaussian_blur=0):
        print(f'<theta_slices> Generating {len(theta_vals)} theta slices...')
        volume = arr
        theta_indexes = []
        print(f'<theta_slices> {volume.shape}')
        for theta in theta_vals:
            # for ioi in [0, 200, 401, ]:
            ioi = int((theta / 180.0) * self.nth)
            print(f'<theta_slices> {theta}° - {ioi} index')
            theta_indexes.append(ioi)
            disp = volume[:, :, ioi]
            if gaussian_blur > 0:
                disp = sp_gaussian_filter(disp, sigma=gaussian_blur)
            plt.figure()
            plt.title(f'{self.tag} - theta = {theta}')
            plt.imshow(
                disp[:int((self.r_plim / self.rmax) * self.nr),
                :int((self.r_plim / self.rmax) * self.nr)],
                extent=[0, self.r_plim, 0, self.r_plim],
                origin='lower',
                cmap=cmap
            )
            plt.clim(clims[0], clims[1])
            plt.colorbar()
            plt.tight_layout()
            plt.xlabel("r (nm)")
            plt.ylabel("r' (nm)")
        if show:
            plt.show()

    """
    WARNING: IN DEVELOPMENT
    """

    def generic_position_blur(self, sigma=1, arr='', kernal_size=3):
        # first build the smoothing kernel
        sigma = sigma  # width of kernel
        x = np.arange(-kernal_size, kernal_size + 1, 1)  # coordinate arrays -- make sure they contain 0!
        y = np.arange(-kernal_size, kernal_size + 1, 1)
        z = np.arange(-kernal_size, kernal_size + 1, 1)
        xx, yy, zz = np.meshgrid(x, y, z)
        kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

        # apply to sample data

        # plt.imshow(arr[:, :, 30])

        filtered = signal.convolve(arr, kernel, mode="same")
        # self.disp_padf = filtered
        arr = filtered
        # plt.figure()
        # plt.imshow(arr[:, :, 30])
        # plt.show()
        print(arr)
        return filtered

    def volume_integration(self, r_range=(0, 1), th_range=(0, 1), arr=''):
        print(f'<volume_integration> Integrating {r_range} Å, {th_range}°')
        # First, normalise the whole volume
        volume = np.array(arr)
        volume = self.padf
        volume_sum = np.sum(volume)
        print(f'volume sum :: {volume_sum}')
        volume = volume / volume_sum

        # Second, find the indexes for the ranges
        r_yard_stick = np.arange(self.r_dist_bin, self.rmax + self.r_dist_bin, self.r_dist_bin)
        th_yard_stick = np.arange(0, 180.0, self.angular_bin)
        r_min_index = (np.abs(r_yard_stick - r_range[0])).argmin()
        r_max_index = (np.abs(r_yard_stick - r_range[1])).argmin()
        th_min_index = (np.abs(th_yard_stick - th_range[0])).argmin()
        th_max_index = (np.abs(th_yard_stick - th_range[1])).argmin()
        print(f'{r_range[0]} :: {r_min_index}')
        print(f'{r_range[1]} :: {r_max_index}')
        print(f'{th_range[0]} :: {th_min_index}')
        print(f'{th_range[1]} :: {th_max_index}')
        # Sum up the voxels within those ranges
        subvolume_sum = np.sum(volume[r_min_index:r_max_index, r_min_index:r_max_index, th_min_index:th_max_index])
        print(f'subvolume_sum :: {self.title} {subvolume_sum}')
        return subvolume_sum
