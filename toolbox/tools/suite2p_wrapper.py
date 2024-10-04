from glob import glob
import isx
import logging
import numpy as np
import os
import shutil
import suite2p
from tarfile import TarFile
from toolbox.utils import io, metadata, preview, utilities
import xml.etree.ElementTree as ET
from zipfile import ZipFile

# from toolbox.tools.process_file import process_file


logger = logging.getLogger()


def run_suite2p_end_to_end(
    raw_movie_files,
    ops_file=None,
    classifier_path=None,
    params_from="table",
    tau=1.0,
    frames_include=-1,
    save_npy=True,
    save_isxd=True,
    save_NWB=False,
    save_mat=False,
    maxregshift=0.1,
    th_badframes=1.0,
    nonrigid=True,
    threshold_scaling=1.0,
    neucoeff=0.7,
    thresh_spks_perc=99.7,
    viz_vmin_perc=0,
    viz_vmax_perc=99,
    viz_cmap="plasma",
    viz_show_grid=True,
    viz_ticks_step=128,
    viz_display_rate=10,
    viz_n_samp_cells=20,
    viz_random_seed=0,
    viz_show_all_footprints=True,
):
    """
    Tool to run end-to-end suite2p pipeline on Inscopix isxd or Bruker Ultima 2P movies.
    """
    # initialize suite2p parameters
    if ops_file is None:
        # load default parameters (user-defined parameters are included afterwards)
        ops = suite2p.default_ops()
    else:
        # load custom parameters (overwrite those of the Analysis table provided to this function as input arguments)
        logger.info(f"Loading custom parameter file {ops_file}.")
        ops_ext = os.path.splitext(ops_file[0])[-1]
        if ops_ext == ".npy":
            ops = np.load(ops_file[0], allow_pickle=True).item()
            if params_from == "file":
                logger.info(
                    "`ops_file` provided: overwriting input parameters from the analysis table."
                )
                tau = ops["tau"]
                frames_include = ops["frames_include"]
                save_NWB = ops["save_NWB"]
                save_mat = ops["save_mat"]
                maxregshift = ops["maxregshift"]
                th_badframes = ops["th_badframes"]
                nonrigid = ops["nonrigid"]
                threshold_scaling = ops["threshold_scaling"]
                neucoeff = ops["neucoeff"]
                classifier_path = ops["classifier_path"]
        else:
            raise ValueError(
                f"'{ops_ext}' is not a supported suite2p parameter file extension. Please provide a NumPy .npy file."
            )

    # get extension of input movie and relevant metadata
    file_ext = "." + ".".join(
        os.path.basename(raw_movie_files[0]).split(".")[1:]
    )
    if file_ext == ".isxd":
        logger.info(
            "Inscopix .isxd movie(s) detected: setting `ops['isxd']` to `True`."
        )
        fs = 1e6 / isx.Movie.read(raw_movie_files[0]).timing.period.to_usecs()
        ops["isxd"] = True
    elif file_ext in [".zip", ".tar.gz"]:
        logger.info(
            f"Bruker Ultima 2P {file_ext} movie(s) detected: setting `ops['bruker']` to `True`."
        )
        data_dir = "/ideas/data/tmp/"
        os.makedirs(data_dir, mode=0o777, exist_ok=True)
        for raw_movie_file in raw_movie_files:
            if file_ext == ".zip":
                with ZipFile(raw_movie_file, "r") as f:
                    for member_info in f.infolist():
                        if member_info.is_dir():
                            continue
                        member_info.filename = os.path.basename(
                            member_info.filename
                        )
                        f.extract(member_info, data_dir)
            elif file_ext == ".tar.gz":
                with TarFile.open(raw_movie_file, "r") as f:
                    for member_info in f.getmembers():
                        if member_info.isdir():
                            continue
                        member_info.name = os.path.basename(member_info.name)
                        f.extract(member_info, data_dir)

        # read XML to get version and fs
        xml_file = glob(data_dir + "*.xml")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bruker_version = root.attrib.get("version")
        fs = 1 / float(
            root.findall('.//PVStateValue/[@key="framePeriod"]')[0].attrib.get(
                "value"
            )
        )
        print(f"Processing Bruker 2p data v{bruker_version}...")
        ops["bruker"] = True
    elif file_ext in [".tif", ".tiff"]:
        fs = ops["fs"]
    else:
        raise ValueError(
            f"File format {file_ext} not recognized as either Inscopix .isxd, Bruker Ultima 2P .zip or .tar.gz, or standard .tif/.tiff stack."
        )

    # Set user-defined parameters
    ops["fs"] = float(fs)
    ops["tau"] = tau
    ops["keep_movie_raw"] = True
    ops["save_mat"] = save_mat
    ops["save_NWB"] = save_NWB
    ops["frames_include"] = frames_include
    ops["maxregshift"] = maxregshift
    ops["th_badframes"] = th_badframes
    ops["nonrigid"] = nonrigid
    ops["threshold_scaling"] = threshold_scaling
    ops["neucoeff"] = neucoeff
    if classifier_path is not None:
        ops["classifier_path"] = classifier_path[0]

    # Set hardcoded parameters
    ops = utilities.set_hardcoded_parameters(ops)
    ideas_output_dir = ops["save_folder"]

    # define directory containing the input movie(s)
    if "data_dir" not in locals():
        data_dir = os.path.dirname(raw_movie_files[0])
    db = {
        "data_path": [data_dir],
    }

    # run pipeline
    output_ops = suite2p.run_s2p(ops=ops, db=db)
    suite2p_output_dir = os.path.dirname(output_ops["ops_path"])

    # output preview(s)
    preview.create_output_previews(
        ops=output_ops,
        steps="all",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
        display_rate=int(viz_display_rate),
        n_samp_cells=int(viz_n_samp_cells),
        thresh_spks_perc=thresh_spks_perc,
        random_seed=int(viz_random_seed),
        show_all_footprints=viz_show_all_footprints,
    )

    # handle output file(s)
    if save_npy:
        # zip suite2p outputs
        zip_output_file = f"{ideas_output_dir}/suite2p_output.zip"
        fname_list = ["F", "Fneu", "iscell", "ops", "spks", "stat"]
        with ZipFile(zip_output_file, "w") as f:
            for fname in fname_list:
                f.write(f"{suite2p_output_dir}/{fname}.npy")
    if save_isxd:
        io.npy_to_isxd(
            npy_dir=suite2p_output_dir,
            output_dir=ideas_output_dir,
            thresh_spks_perc=thresh_spks_perc,
        )
    if save_mat:
        mat_output_file = f"{suite2p_output_dir}/Fall.mat"
        shutil.move(mat_output_file, ideas_output_dir)

    # output metadata
    metadata.create_output_metadata(
        ops=output_ops,
        steps="all",
    )

    # clean up suite2p output folder (otherwise recognized as output by IDEAS)
    shutil.rmtree(suite2p_output_dir)
    print("ALL DONE!")
