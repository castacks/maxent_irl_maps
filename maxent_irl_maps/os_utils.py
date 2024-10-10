"""
set of utility functions for file management
"""

import os
import numpy as np


def maybe_mkdir(fp, force=True):
    if not os.path.exists(fp):
        os.makedirs(fp)
    elif not force:
        x = input(
            "{} already exists. Hit enter to continue and overwrite. Q to exit.".format(
                fp
            )
        )
        if x.lower() == "q":
            exit(0)


def walk_bags(fp, extension=".bag"):
    """
    Args:
        fp: The base directory to walk
        extension: The extension to look for

    Returns:
        An list of the full fps for all files rooted at fp with extension extension
    """
    res = []
    ctimes = []
    ext_len = len(extension)

    for root, dirs, files in os.walk(fp):
        for f in files:
            if f[-ext_len:] == extension:
                fp_new = os.path.join(root, f)
                res.append(fp_new)
                ctimes.append(os.path.getctime(fp_new))

    res = np.array(res)
    ctimes = np.array(ctimes)

    return list(res[np.argsort(ctimes)])
