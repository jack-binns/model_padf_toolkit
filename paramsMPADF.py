from params_new import params
import os

"""
Class to set up the model PADF parameters
"""


class paramsMPADF(params):

    def __init__(self):
        ch = ["model PADF"]

        params.__init__(self, "Model PADF parameters", configheaders=ch)

        self.add_parameter("root", "None", cmdline="--root", help="Parent root",
                           nargs=1, header=ch[0], pathflag=True)

        self.add_parameter("project", "None", cmdline="--project",
                           help="Path where files will be written. Can be blank",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("tag", "None", cmdline="--tag", help="text string prefix for each output file.",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("nr", int(100), cmdline="--nr", help="Number of real-space radial samples",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("nth", int(100), cmdline="--nth", help="Number of real-space angular samples",
                           nargs=1, header=ch[0], pathflag=False)

        self.add_parameter("rmax", 10.0, cmdline="--rmax", help="maximum r value (A)",
                           nargs=1, header=ch[0], pathflag=False)

    def read_calc_param_file(self, fname):
        print("<read_calc_param_file> Opening config file name....    :  ", fname)
        self.read_parameters_from_file(fname)
        print("<read_calc_param_file> ...read config file", fname)
