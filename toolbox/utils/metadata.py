import json
import logging
from pathlib import Path

import numpy as np

from toolbox.utils.nwb_utils import construct_nwb_file_metadata

logger = logging.getLogger()


def create_output_metadata(ops, steps, efocus_vals=None):
    """
    Creates metadata for output files of the suite2p pipeline and individual steps tools
    """
    L = ops["nframes"]

    metadata_general_dict = {
        "Number of Frames": L,
        "Sampling Rate (Hz)": round(ops["fs"], 2),
        "Recording Duration (s)": round(L / ops["fs"], 3),
        "Number of Planes": ops["nplanes"],
        "Number of Channels": ops["nchannels"],
        "Frame Width (px)": ops["Lx"],
        "Frame Height (px)": ops["Ly"],
    }

    npy_dir = ops["save_path"]
    if steps in ["all", "output_conversion"]:
        npy_output_name = "suite2p_output.zip"
        cs_output_name = "cellset_raw.isxd"
        es_output_name = "eventset.isxd"

        npy_output_key = Path(npy_output_name).stem
        cs_output_key = Path(cs_output_name).stem
        es_output_key = Path(es_output_name).stem

        F = np.load(f"{npy_dir}/F.npy", allow_pickle=True)
        spks = np.load(f"{npy_dir}/spks.npy", allow_pickle=True)
        stat = np.load(f"{npy_dir}/stat.npy", allow_pickle=True)
        iscell = np.load(f"{npy_dir}/iscell.npy", allow_pickle=True)

        # if allow_overlap is False, then fluorescence traces of overlapping footprints are set to all zeros; the code below removes data from overlapping ROIs from preview figures
        if not ops["extraction"]["allow_overlap"]:
            idx_ok = np.where([len(np.unique(f)) > 1 for f in F])[0]
            F = F[idx_ok, :]
            spks = spks[idx_ok, :]
            stat = stat[idx_ok]
            iscell = iscell[idx_ok]

        F = F[:, :L]
        spks = spks[:, :L]

        avg_spk_rates = np.mean((spks[iscell[:, 0] == 1, :] > 1), axis=1)

        extra_dict = {
            "Number of Extracted ROIs": len(iscell),
            "Number of Accepted ROIs": int(np.sum(iscell[:, 0])),
            "Number of Rejected ROIs": int(np.sum(iscell[:, 0] == 0)),
            "Average ROI Radius (px)": round(
                float(np.mean([x["radius"] for x in stat])), 2
            ),
            "Average trace amplitude": round(
                float(np.mean(F[iscell[:, 0] == 1, :])), 3
            ),
            "Average event rate (Hz)": round(float(np.mean(avg_spk_rates)), 4),
            "Minimum event rate (Hz)": round(float(np.min(avg_spk_rates)), 4),
            "Maximum event rate (Hz)": round(float(np.max(avg_spk_rates)), 4),
        }

        dict_items = list(metadata_general_dict.items())
        dict_items.insert(0, ("Cell Extraction Method", "suite2p"))
        metadata_output_dict = dict(dict_items)
        metadata_output_dict.update(extra_dict)

        cs_keys_list = [
            "Cell Extraction Method",
            "Number of Frames",
            "Sampling Rate (Hz)",
            "Recording Duration (s)",
            "Number of Extracted ROIs",
            "Number of Accepted ROIs",
            "Number of Rejected ROIs",
            "Average trace amplitude",
        ]
        metadata_cs_output_dict = {
            key: metadata_output_dict.get(key, "") for key in cs_keys_list
        }
        metadata_cs_output_dict["microscope"] = {"focus": efocus_vals}

        es_keys_list = [
            "Cell Extraction Method",
            "Number of Frames",
            "Sampling Rate (Hz)",
            "Recording Duration (s)",
            "Average event rate (Hz)",
            "Minimum event rate (Hz)",
            "Maximum event rate (Hz)",
        ]
        metadata_es_output_dict = {
            key: metadata_output_dict.get(key, "") for key in es_keys_list
        }
        metadata_es_output_dict["microscope"] = {"focus": efocus_vals}

        metadata = {
            npy_output_key: metadata_output_dict,
            cs_output_key: metadata_cs_output_dict,
            es_output_key: metadata_es_output_dict,
        }

        if ops["io"]["save_NWB"]:
            nwb_output_name = "ophys.nwb"
            nwb_output_key = Path(nwb_output_name).stem
            try:
                nwb_file_metadata = construct_nwb_file_metadata(nwb_output_name)
                nwb_file_metadata.update(metadata_output_dict)
            except Exception as e:
                logger.warning(f"Could not construct NWB file metadata: {str(e)}")
                nwb_file_metadata = metadata_output_dict
            metadata.update({nwb_output_key: nwb_file_metadata})

        if ops["io"]["save_mat"]:
            mat_output_name = "Fall.mat"
            mat_output_key = Path(mat_output_name).stem
            metadata.update({mat_output_key: metadata_output_dict})

        if steps == "all":
            img_output_name = "local_corr_img.tif"
            img_output_key = Path(img_output_name).stem
            img_keys_list = [
                "Frame Width (px)",
                "Frame Height (px)",
            ]
            metadata_img_output_dict = {
                key: metadata_output_dict.get(key, "") for key in img_keys_list
            }
            metadata.update({img_output_key: metadata_img_output_dict})

    elif steps in ["binary_conversion", "registration"]:
        if steps == "binary_conversion":
            bin_output_name = "data_raw.bin"
        else:
            bin_output_name = "data.bin"
        bin_output_key = Path(bin_output_name).stem
        metadata = {bin_output_key: metadata_general_dict}

    elif steps in ["roi_detection", "roi_extraction"]:
        if steps == "roi_detection":
            stat_output_name = "stat_ROI_detection.npy"
        else:
            stat_output_name = "stat.npy"

        stat = np.load(f"{npy_dir}/{stat_output_name}", allow_pickle=True)

        stat_output_dict = {
            "Number of Extracted ROIs": len(stat),
            "Average ROI Radius (px)": round(
                float(np.mean([x["radius"] for x in stat])), 2
            ),
        }

        if steps == "roi_extraction":
            F = np.load(f"{npy_dir}/F.npy", allow_pickle=True)
            extra_dict = {
                "Average trace amplitude": round(float(np.mean(F)), 3),
            }
            stat_output_dict.update(extra_dict)

        dict_items = list(metadata_general_dict.items())
        dict_items.insert(0, ("Cell Extraction Method", "suite2p"))
        metadata_output_dict = dict(dict_items)
        metadata_output_dict.update(stat_output_dict)

        stat_output_key = Path(stat_output_name).stem
        metadata = {stat_output_key: metadata_output_dict}

    elif steps == "roi_classification":
        iscell = np.load(f"{npy_dir}/iscell.npy", allow_pickle=True)
        extra_dict = {
            "Number of Extracted ROIs": len(iscell),
            "Number of Accepted ROIs": int(np.sum(iscell[:, 0])),
            "Number of Rejected ROIs": int(np.sum(iscell[:, 0] == 0)),
        }

        dict_items = list(metadata_general_dict.items())
        dict_items.insert(0, ("Cell Extraction Method", "suite2p"))
        metadata_output_dict = dict(dict_items)
        metadata_output_dict.update(extra_dict)

        iscell_output_key = Path("iscell.npy").stem
        metadata = {iscell_output_key: metadata_output_dict}

    elif steps == "spike_deconvolution":
        spks = np.load(f"{npy_dir}/spks.npy", allow_pickle=True)
        spks = spks[:, :L]

        avg_spk_rates = np.mean((spks > 1), axis=1)

        extra_dict = {
            "Number of Extracted ROIs": spks.shape[0],
            "Average event rate (Hz)": round(float(np.mean(avg_spk_rates)), 4),
            "Minimum event rate (Hz)": round(float(np.min(avg_spk_rates)), 4),
            "Maximum event rate (Hz)": round(float(np.max(avg_spk_rates)), 4),
        }

        dict_items = list(metadata_general_dict.items())
        dict_items.insert(0, ("Cell Extraction Method", "suite2p"))
        metadata_output_dict = dict(dict_items)
        metadata_output_dict.update(extra_dict)

        spks_output_key = Path("spks.npy").stem
        metadata = {spks_output_key: metadata_output_dict}

    with open("output_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
