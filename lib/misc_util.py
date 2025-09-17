"""
Miscellaneous utility functions likely used in more than one file.
"""

import pickle

def split_trans_method(d):
    """
    Split values in study parameters to differnt dicts by namespace
    """
    tnamespace = "trans::"
    mnamespace = "method::"
    tdict = dict()
    tdict_pars = dict()
    mdict = dict()

    # transform
    for key,value in d.items():
        if key.startswith(tnamespace):
            spl = key.split(tnamespace)[1]
            if spl == "type":
                tdict["name"] = value
            else:
                tdict_pars[spl] = value
    tdict["parameters"] = tdict_pars

    # method
    for key,value in d.items():
        if key.startswith(mnamespace):
            spl = key.split(mnamespace)[1]
            mdict[spl] = value

    return tdict,mdict


def open_bundle(filename):
    with open(filename, "rb") as f:
        loaded_bundle = pickle.load(f)
    loaded_bundle["load_path"] = filename
    return loaded_bundle
