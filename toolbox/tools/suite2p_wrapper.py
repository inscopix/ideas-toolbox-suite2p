import os
import shutil
from typing import List, Optional
from zipfile import ZipFile

import isx
import numpy as np
import suite2p
from ideas.tools import log
from ideas.tools.types import IdeasFile
from ideas.tools import outputs

from toolbox.utils import io, metadata, preview, utilities

logger = log.get_logger()


def run_suite2p_end_to_end(
    *,
    raw_movie_files: List[str],
    ops_file: Optional[List[str]] = None,
    classifier_path: Optional[List[str]] = None,
    params_from: bool = False,
    tau: float = 1.0,
    frames_include: int = -1,
    save_npy: bool = True,
    save_isxd: bool = True,
    save_NWB: bool = False,
    save_mat: bool = False,
    save_img: bool = True,
    align_by_chan: int = 1,
    maxregshift: float = 0.1,
    th_badframes: float = 1.0,
    nonrigid: bool = True,
    threshold_scaling: float = 1.0,
    neucoeff: float = 0.7,
    thresh_spks_perc: float = 99.7,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_display_rate: float = 10.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Tool to run end-to-end suite2p pipeline on Inscopix isxd or Bruker Ultima 2P movies.

    :param List[str] raw_movie_files: Input 2P movie(s).
    :param Optional[List[str]] ops_file: Optional parameters file that allows for more granular control of all available suite2p parameters. See documentation for more details.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param bool params_from: When a parameters file is provided as input, whether the parameters from the analysis table columns should be overwritten by the values from the file. Only effective when an optional parameters file is provided. Defaults to False.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param int frames_include: [from suite2p docs] If greater than zero, only [the first] <frames_include> frames are processed. Useful for testing parameters on a subset of data.
    :param bool save_npy: If true, save suite2p NPY output (as a ZIP file).
    :param bool save_isxd: If true, save suite2p output as ISXD files.
    :param bool save_NWB: [from suite2p docs] Whether to save output as NWB file.
    :param bool save_mat: [from suite2p docs] Whether to save the results in matlab format in file "Fall.mat".
    :param bool save_img: Whether to save the local correlation image as a standalone .tif file, e.g., for further use as template image in Multi-Session Registration.
    :param int align_by_chan: [from suite2p docs] Which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel). If you have a non-functional channel with something like td-Tomato expression, you may want to use this channel for alignment rather than the functional channel.
    :param float maxregshift: [from suite2p docs] The maximum shift as a fraction of the frame size. If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops['maxregshift'].
    :param float th_badframes: [from suite2p docs] Involved with setting threshold for excluding frames for cropping. Set this smaller to exclude more frames.
    :param bool nonrigid: [from suite2p docs] Whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
    :param float threshold_scaling: [from suite2p docs] This controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected). If you set this higher, then fewer ROIs will be detected, and if you set it lower, more ROIs will be detected.
    :param float neucoeff: [from suite2p docs] Neuropil coefficient for all ROIs.
    :param float thresh_spks_perc: Threshold for denoising the deconvolved spike trains, in percentile. Any value in the ROIs-by-time-point deconvolved spike matrix that is below the matrix's xth percentile value is set to 0. Note that the same threshold is applied to all ROIs, and that this thresholding step does not binarize the deconvolved spike trains but simply filters out the low-amplitude spike events.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param float viz_display_rate: Display rate for the preview movies, in Hz.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    # initialize suite2p parameters
    if ops_file is None or len(ops_file) == 0:
        # load default parameters (user-defined parameters are included afterwards)
        ops = suite2p.default_ops()
    else:
        # load custom parameters (overwrite those of the Analysis table provided to this function as input arguments)
        logger.info(f"Loading custom parameter file {ops_file}.")
        ops_ext = os.path.splitext(ops_file[0])[-1]
        if ops_ext == ".npy":
            ops = np.load(ops_file[0], allow_pickle=True).item()
            if params_from:
                logger.info(
                    "`ops_file` provided: overwriting input parameters from the analysis table."
                )
                tau = ops["tau"]
                frames_include = ops["frames_include"]
                save_NWB = ops["save_NWB"]
                save_mat = ops["save_mat"]
                align_by_chan = ops["align_by_chan"]
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
    efocus_vals = [None] * len(raw_movie_files)
    file_ext = "." + ".".join(os.path.basename(raw_movie_files[0]).split(".")[1:])
    if file_ext == ".isxd":
        logger.info(
            "Inscopix .isxd movie(s) detected: setting `ops['isxd']` to `True`."
        )
        movie = isx.Movie.read(raw_movie_files[0])
        fs = 1 / movie.timing.period.secs_float
        start_time = movie.timing.start.to_datetime()
        # converting `start_time` to a NumPy array of dtype np.datetime64, otherwise it cannot be saved as a .mat file
        start_time = np.array(start_time, dtype=np.datetime64)
        ops["isxd"] = True
        efocus_vals = utilities.get_efocus_vals(raw_movie_files)
    elif file_ext in [".zip", ".tar.gz"]:
        logger.info(
            f"Bruker Ultima 2P {file_ext} movie(s) detected: setting `ops['input_format']` to `'tif'`."
        )
        data_dir = "/ideas/data/tmp/"
        os.makedirs(data_dir, mode=0o777, exist_ok=True)
        fs, start_time = io.extract_bruker2p_file(
            raw_movie_files=raw_movie_files,
            file_ext=file_ext,
            data_dir=data_dir,
        )
        ops["input_format"] = "tif"
    elif file_ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff"]:
        fs = ops["fs"]
        start_time = None
    else:
        raise ValueError(
            f"File format {file_ext} not recognized as either Inscopix .isxd, Bruker Ultima 2P .zip or .tar.gz, or standard .tif/.tiff/.ome.tif/.ome.tiff stack."
        )

    # Set user-defined parameters
    ops["fs"] = float(fs)
    ops["start_time"] = start_time
    ops["tau"] = tau
    ops["keep_movie_raw"] = True
    ops["save_mat"] = save_mat
    ops["save_NWB"] = save_NWB
    ops["frames_include"] = frames_include
    ops["align_by_chan"] = align_by_chan
    ops["maxregshift"] = maxregshift
    ops["th_badframes"] = th_badframes
    ops["nonrigid"] = nonrigid
    ops["threshold_scaling"] = threshold_scaling
    ops["neucoeff"] = neucoeff
    if classifier_path is not None:
        if isinstance(classifier_path, List) and len(classifier_path) > 0:
            ops["classifier_path"] = classifier_path[0]
        else:
            ops["classifier_path"] = classifier_path

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
    if save_img:
        img_path = io.save_local_corr_img(
            ops=output_ops,
            output_dir=ideas_output_dir,
        )
        preview.preview_template_image(img_path)

    # output metadata
    metadata.create_output_metadata(
        ops=output_ops, steps="all", efocus_vals=efocus_vals
    )

    # clean up suite2p output folder (otherwise recognized as output by IDEAS)
    shutil.rmtree(suite2p_output_dir)
    print("ALL DONE!")


def run_suite2p_end_to_end_ideas_wrapper(
    *,
    raw_movie_files: List[IdeasFile],
    ops_file: Optional[List[IdeasFile]] = None,
    classifier_path: Optional[List[IdeasFile]] = None,
    params_from: bool = False,
    tau: float = 1.0,
    frames_include: int = -1,
    save_npy: bool = True,
    save_isxd: bool = True,
    save_NWB: bool = False,
    save_mat: bool = False,
    save_img: bool = True,
    align_by_chan: int = 1,
    maxregshift: float = 0.1,
    th_badframes: float = 1.0,
    nonrigid: bool = True,
    threshold_scaling: float = 1.0,
    neucoeff: float = 0.7,
    thresh_spks_perc: float = 99.7,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_display_rate: float = 10.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Ideas wrapper for tool to run end-to-end suite2p pipeline on Inscopix isxd or Bruker Ultima 2P movies.

    :param List[str] raw_movie_files: Input 2P movie(s).
    :param Optional[List[str]] ops_file: Optional parameters file that allows for more granular control of all available suite2p parameters. See documentation for more details.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param bool params_from: When a parameters file is provided as input, whether the parameters from the analysis table columns should be overwritten by the values from the file. Only effective when an optional parameters file is provided. Defaults to False.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param int frames_include: [from suite2p docs] If greater than zero, only [the first] <frames_include> frames are processed. Useful for testing parameters on a subset of data.
    :param bool save_npy: If true, save suite2p NPY output (as a ZIP file).
    :param bool save_isxd: If true, save suite2p output as ISXD files.
    :param bool save_NWB: [from suite2p docs] Whether to save output as NWB file.
    :param bool save_mat: [from suite2p docs] Whether to save the results in matlab format in file "Fall.mat".
    :param bool save_img: Whether to save the local correlation image as a standalone .tif file, e.g., for further use as template image in Multi-Session Registration.
    :param int align_by_chan: [from suite2p docs] Which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel). If you have a non-functional channel with something like td-Tomato expression, you may want to use this channel for alignment rather than the functional channel.
    :param float maxregshift: [from suite2p docs] The maximum shift as a fraction of the frame size. If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops['maxregshift'].
    :param float th_badframes: [from suite2p docs] Involved with setting threshold for excluding frames for cropping. Set this smaller to exclude more frames.
    :param bool nonrigid: [from suite2p docs] Whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
    :param float threshold_scaling: [from suite2p docs] This controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected). If you set this higher, then fewer ROIs will be detected, and if you set it lower, more ROIs will be detected.
    :param float neucoeff: [from suite2p docs] Neuropil coefficient for all ROIs.
    :param float thresh_spks_perc: Threshold for denoising the deconvolved spike trains, in percentile. Any value in the ROIs-by-time-point deconvolved spike matrix that is below the matrix's xth percentile value is set to 0. Note that the same threshold is applied to all ROIs, and that this thresholding step does not binarize the deconvolved spike trains but simply filters out the low-amplitude spike events.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param float viz_display_rate: Display rate for the preview movies, in Hz.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    run_suite2p_end_to_end(
        raw_movie_files=raw_movie_files,
        ops_file=ops_file,
        classifier_path=classifier_path,
        params_from=params_from,
        tau=tau,
        frames_include=frames_include,
        save_npy=save_npy,
        save_isxd=save_isxd,
        save_NWB=save_NWB,
        save_mat=save_mat,
        save_img=save_img,
        align_by_chan=align_by_chan,
        maxregshift=maxregshift,
        th_badframes=th_badframes,
        nonrigid=nonrigid,
        threshold_scaling=threshold_scaling,
        neucoeff=neucoeff,
        thresh_spks_perc=thresh_spks_perc,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
        viz_display_rate=viz_display_rate,
        viz_n_samp_cells=viz_n_samp_cells,
        viz_random_seed=viz_random_seed,
        viz_show_all_footprints=viz_show_all_footprints,
    )


    try:
        logger.info("Registering output data")
        output_prefix = outputs.input_paths_to_output_prefix(
            raw_movie_files, ops_file, max_name_len=100
        )
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            suite2p_output_file, cellset_raw_file, eventset_file, ophys_file, mat_file = None, None, None, None, None

            if save_npy:
                suite2p_output_file = output_data.register_file(
                    "suite2p_output.zip",
                    prefix=output_prefix
                ).register_metadata_dict(
                    **metadata["suite2p_output"]
                )
            
            if save_isxd:
                cellset_raw_file = output_data.register_file(
                    "cellset_raw.isxd",
                    prefix=output_prefix,
                ).register_metadata_dict(
                    **metadata["cellset_raw"]
                )

                eventset_file = output_data.register_file(
                    "eventset.isxd",
                    prefix=output_prefix,
                ).register_metadata_dict(
                    **metadata["eventset"]
                )

            if save_NWB:
                ophys_file = output_data.register_file(
                    "ophys.nwb",
                    prefix=output_prefix,
                ).register_metadata_dict(
                    **metadata["ophys"]
                )

            if save_mat:
                mat_file = output_data.register_file(
                    "Fall.mat",
                    prefix=output_prefix,
                ).register_metadata_dict(
                    **metadata["Fall"]
                )

            if save_img:
                output_data.register_file(
                    "local_corr_img.tif",
                    prefix=output_prefix
                ).register_preview(
                    "local_corr_img_preview.png",
                    caption="Local correlation image (from local_corr_img.tif)",
                    prefix=""
                )

            for f in [suite2p_output_file, ophys_file, mat_file]:
                if not f:
                    continue
                f.register_preview(
                    "registration_fovs.svg",
                    caption="Various FOVs from the registration process (from ops.npy)",
                    prefix=""
                ).register_preview(
                    "registration_offsets.svg",
                    caption="x and y offsets for both rigid and non-rigid registration (from ops.npy)",
                    prefix=""
                ).register_preview(
                    "registration_movies.mp4",
                    caption="Side-by-side raw and registered movies (from ops.npy, data_raw.bin, and data.bin)",
                    prefix=""
                ).register_preview(
                    "detection_footprints_all.svg",
                    caption="Various FOVs from the ROI detection process (from ops.npy, stat.npy, and iscell.npy)",
                    prefix=""
                ).register_preview(
                    "detection_footprints_accepted.svg",
                    caption="FOV of the accepted ROIs from the ROI detection process (from ops.npy, stat.npy, and iscell.npy)",
                    prefix=""
                ).register_preview(
                    "extracted_sample_sources_traces_suite2p.svg",
                    caption="Sample fluorescence traces, neuropil traces and deconvolved spikes (from ops.npy, F.npy, Fneu.npy, spks.npy, and iscell.npy)",
                    prefix=""
                )

            for f in [suite2p_output_file, ophys_file, mat_file, cellset_raw_file]:
                if not f:
                    continue
                f.register_preview(
                    "extracted_sample_sources_traces_only.svg",
                    caption="Sample fluorescence traces only (from ops.npy, F.npy, and iscell.npy)",
                    prefix=""
                ).register_preview(
                    "extracted_sample_sources_footprints.svg",
                    caption="Footprints of the sample sources (from ops.npy, stat.npy, and iscell.npy)",
                    prefix=""
                )

            for f in [suite2p_output_file, ophys_file, mat_file, eventset_file]:
                if not f:
                    continue
                f.register_preview(
                    "extracted_sample_sources_traces_spikes.svg",
                    caption="Sample fluorescence traces and deconvolved spikes (from ops.npy, F.npy, spks.npy, and iscell.npy)",
                    prefix=""
                ).register_preview(
                    "raster_deconvolved_spikes.svg",
                    caption="Raster plot of the deconvolved spikes (from ops.npy, spks.npy, and iscell.npy)",
                    prefix=""
                )
                
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")