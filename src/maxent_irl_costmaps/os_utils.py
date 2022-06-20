"""
set of utility functions for file management
"""

import os
import numpy as np

def walk_bags(fp, extension='.bag'):
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
        for f  in files:
            if f[-ext_len:] == extension:
                fp_new = os.path.join(root, f)
                res.append(fp_new)
                ctimes.append(os.path.getctime(fp_new))

    res = np.array(res)
    ctimes = np.array(ctimes)

    return list(res[np.argsort(ctimes)])
