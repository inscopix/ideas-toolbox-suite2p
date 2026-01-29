import os
from ideas.analysis.metadata import get_efocus


def set_hardcoded_parameters(ops):
    """Set the hardcoded parameters in the ops dictionary, as a separate function to reduce redundancy between the different suite2p tools."""
    ops["force_sktiff"] = True
    ops["multiplane_parallel"] = False
    ops["fast_disk"] = []
    ops["delete_bin"] = False
    ops["nwb_driver"] = ""
    ops["save_path0"] = ""
    ops["look_one_level_down"] = False
    ops["subfolders"] = []
    ops["move_bin"] = False
    ops["combined"] = True
    ops["keep_movie_raw"] = True
    ops["reg_tif"] = False
    ops["reg_tif_chan2"] = False
    ops["roidetect"] = True
    ops["save_folder"] = os.getcwd()
    return ops


def get_efocus_vals(raw_movie_files):
    """Extract efocus values from input ISXD movies"""
    efocus_vals = [None] * len(raw_movie_files)
    for idx, f in enumerate(raw_movie_files):
        if f.lower().endswith(".isxd"):
            try:
                efocus_vals[idx] = get_efocus(f)
            except Exception:
                pass
    return efocus_vals
