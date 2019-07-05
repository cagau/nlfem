#-*- coding:utf-8 -*-

import numpy as np
import datetime
#from time import strftime

def filename(mesh_name, delta, Tstmp=False):
    """
    Generates filename for given input.

    :param mesh_name: Name of mesh, e.g. "medium".
    :param delta: Interaction radius.
    :param Tstmp: bool, If True a timestamp will computed, otherwise it is set to "".
    :return: list of string. Tstmp, filename
    """
    if Tstmp:
        Tstmp = timestamp()
    else:
        Tstmp = ""
    return Tstmp, mesh_name + str(round(delta * 10))

def timestamp():
    """
    Returns current timestamp as string.

    :return: string, format %m%d_%H-%M-%S
    """
    # Link to strftime Doc
    # http://strftime.org/
    return datetime.datetime.now().strftime("%m%d_%H-%M-%S")
