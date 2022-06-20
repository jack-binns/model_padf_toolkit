from params_new import params
import os


#
# class to set up the PADF parameters
#


class paramsFILT(params):

    def __init__(self, outpath="None", tag="None", nl=6, nlmin=0, nr=100, nth=100, nq=100,
                 qmin=0.0, qmax=1e9, rmax=1e-7, wl=1e-10):
        ch = ["PADF"]

        params.__init__(self, "PADF parameters", configheaders=ch)

        self.add_parameter("config", "None", cmdline="--config", help="Name of file with all input parameters",
                           nargs=1, header=["PADF"], pathflag=True)

        self.add_parameter("outpath", "None", cmdline="--outpath", help="Path where files will be written.",
                           nargs=1, header=ch[0], pathflag=True)
        self.add_parameter("tag", "None", cmdline="--tag", help="text string prefix for each output file.",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("nl", int(10), cmdline="--nl", help="Number of spherical harmonic values (l)",
                           nargs=1, header=ch[0], pathflag=False)
        self.add_parameter("nlmin", int(0), cmdline="--nlmin", help="Minimum spherical harmonic value l",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("nr", int(100), cmdline="--nr", help="Number of real-space radial samples",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("nq", int(100), cmdline="--nq", help="Number of q-space radial samples",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("qmin", 0.0, cmdline="--qmin", help="minimum q value (A^-1)",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("qmax", 10.0, cmdline="--qmax", help="maximum q value (A^-1)",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("rmax", 10.0, cmdline="--rmax", help="maximum r value (A)",
                           nargs=1, header=ch[0], pathflag=False)

    def read_config_file(self, fname):
        print("<read_config_file> Opening config file name....    :  ", fname)
        self.read_parameters_from_file(fname)
        print("<read_config_file> ...read config file")
