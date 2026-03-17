import os
import shutil
import time
from glob import glob
from typing import List, Optional
from zipfile import ZipFile

import isx
import numpy as np
from ideas.tools import log
from ideas.tools.types import IdeasFile
from ideas.tools import outputs
from natsort import natsorted
from suite2p import (
    classification,
    default_ops,
    detection,
    extraction,
    io,
    registration,
)

from toolbox.utils import io as tlbxio
from toolbox.utils import metadata, preview, utilities

logger = log.get_logger()


def suite2p_binary_conversion(
    *,
    raw_movie_files: List[str],
    nplanes: int = 1,
    nchannels: int = 1,
    functional_chan: int = 1,
    fs: Optional[float] = None,
    bruker_bidirectional: bool = False,
):
    """
    Tool to convert raw 2P input movie(s) into a suite2p binary file. This constitutes the first step of the suite2p end-to-end pipeline.

    :param List[str] raw_movie_files: Input 2P movie(s) [.isxd, .zip, .tiff].
    :param int nplanes: [from suite2p docs] Each tiff has this many planes in sequence.
    :param int nchannels: [from suite2p docs] Each tiff has this many channels per plane.
    :param int functional_chan: [from suite2p docs] This channel is used to extract functional ROIs (1-based, so 1 means first channel, and 2 means second channel).
    :param Optional[float] fs: [from suite2p docs] Sampling rate (per plane). For instance, if you have a 10 plane recording acquired at 30Hz, then the sampling rate per plane is 3Hz, so set ops['fs'] = 3.
    :param bool bruker_bidirectional: [from suite2p docs] Specifies whether BRUKER files are bidirectional multiplane recordings. The True setting corresponds to the following plane order (first plane is indexed as zero): [0,1,2,2,1,0]. False corresponds to [0,1,2,0,1,2].
    """
    t0 = time.time()

    # initialize suite2p parameters
    ops = default_ops()

    # set hardcoded parameters
    ops = utilities.set_hardcoded_parameters(ops)

    # set user-defined parameters
    ops["nplanes"] = nplanes
    ops["nchannels"] = nchannels
    ops["functional_chan"] = functional_chan
    ops["bruker_bidirectional"] = bruker_bidirectional

    # define directory containing the input movie(s)
    data_dir = os.path.dirname(raw_movie_files[0])

    # curate parameters
    if (
        isinstance(ops["diameter"], list)
        and len(ops["diameter"]) > 1
        and ops["aspect"] == 1.0
    ):
        ops["aspect"] = ops["diameter"][0] / ops["diameter"][1]

    # check if there are binaries already made
    if "save_folder" not in ops or len(ops["save_folder"]) == 0:
        ops["save_folder"] = "suite2p"
    save_folder = os.path.join(ops["save_path0"], ops["save_folder"])
    os.makedirs(save_folder, exist_ok=True)
    plane_folders = natsorted(
        [
            f.path
            for f in os.scandir(save_folder)
            if f.is_dir() and f.name[:5] == "plane"
        ]
    )

    # detect file type
    file_ext = "." + ".".join(os.path.basename(raw_movie_files[0]).split(".")[1:])
    if file_ext == ".isxd":
        ops["input_format"] = "isxd"
        movie = isx.Movie.read(raw_movie_files[0])
        fs_auto = 1 / movie.timing.period.secs_float
        start_time = movie.timing.start.to_datetime()
        # converting `start_time` to a NumPy array of dtype np.datetime64, otherwise it cannot be saved as a .mat file
        start_time = np.array(start_time, dtype=np.datetime64)
    elif file_ext in [".zip", ".tar.gz"]:
        logger.info(
            f"Bruker Ultima 2P {file_ext} movie detected: setting `ops['input_format']` to `'tif`."
        )
        data_dir = "/ideas/data/tmp/"
        os.makedirs(data_dir, mode=0o777, exist_ok=True)
        fs_auto, start_time = tlbxio.extract_bruker2p_file(
            raw_movie_files=raw_movie_files,
            file_ext=file_ext,
            data_dir=data_dir,
        )
        ops["input_format"] = "tif"
        ops["force_sktiff"] = True
        tif_list = glob(f"{data_dir}/*tif")
        channel_list = [int(os.path.basename(x).split("_")[2][-1]) for x in tif_list]
        ops["functional_chan"] = channel_list[0]
    elif file_ext in [".tif", ".tiff", ".ome.tif", ".ome.tiff"]:
        ops["input_format"] = "tif"
        fs_auto = ops["fs"]
        start_time = None
    else:
        raise ValueError(
            f"File format {file_ext} not recognized as either Inscopix .isxd, Bruker Ultima 2P .zip or .tar.gz, or standard .tif/.tiff/.ome.tif/.ome.tiff stack."
        )

    # curate parameters
    if fs is not None:
        ops["fs"] = fs
    else:
        ops["fs"] = fs_auto
    ops["start_time"] = start_time
    ops["data_path"] = [data_dir]
    if "save_path0" not in ops or len(ops["save_path0"]) == 0:
        if ops.get("h5py"):
            ops["save_path0"] = os.path.split(ops["h5py"][0])[
                0
            ]  # Use first element in h5py key to find save_path
        elif ops.get("nwb_file"):
            ops["save_path0"] = os.path.split(ops["nwb_file"])[0]
        else:
            ops["save_path0"] = ops["data_path"][0]

    # map file type to conversion function
    convert_funs = {
        "isxd": io.isxd_to_binary,
        "bruker": io.ome_to_binary,
        "tif": io.tiff_to_binary,
    }

    # convert input movie to binary file
    ops0 = convert_funs[ops["input_format"]](ops.copy())
    if isinstance(ops, list):
        ops0 = ops0[0]

    plane_folders = natsorted(
        [
            f.path
            for f in os.scandir(save_folder)
            if f.is_dir() and f.name[:5] == "plane"
        ]
    )
    # ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    logger.info(
        "time {:0.2f} sec. Wrote {} frames per binary for {} planes".format(
            time.time() - t0, ops0["nframes"], len(plane_folders)
        )
    )

    # output metadata
    metadata.create_output_metadata(
        ops=ops0,
        steps="binary_conversion",
    )

    # output previews
    preview.create_output_previews(
        ops=ops0,
        steps="binary_conversion",
    )

    # move output files into output folder
    ideas_output_dir = os.getcwd()
    shutil.move(ops0["raw_file"], f"{ideas_output_dir}/data_raw.bin")
    shutil.move(ops0["ops_path"], f"{ideas_output_dir}/ops_binary_conversion.npy")
    if os.path.exists(f"{ideas_output_dir}/suite2p/"):
        shutil.rmtree(f"{ideas_output_dir}/suite2p/")
    logger.info("ALL DONE!")


def suite2p_registration(
    *,
    raw_binary_file: List[str],
    ops_file: List[str],
    frames_include: int = -1,
    align_by_chan: int = 1,
    nimg_init: int = 300,
    batch_size: int = 500,
    maxregshift: float = 0.1,
    smooth_sigma: float = 1.15,
    smooth_sigma_time: float = 0.0,
    two_step_registration: bool = False,
    subpixel: int = 10,
    th_badframes: float = 1.0,
    norm_frames: bool = True,
    force_refImg: bool = False,
    pad_fft: bool = False,
    one_p_reg: bool = False,
    spatial_hp_reg: int = 42,
    pre_smooth: float = 0.0,
    spatial_taper: float = 40.0,
    nonrigid: bool = True,
    block_size: List[int] = [128, 128],
    snr_thresh: float = 1.2,
    maxregshiftNR: float = 5.0,
    do_bidiphase: bool = False,
    bidiphase: int = 0,
    bidi_corrected: bool = False,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_display_rate: float = 10.0,
):
    """
    Tool to run suite2p registration on a raw suite2p binary movie. This constitutes the second step of the suite2p end-to-end pipeline.

    :param List[str] raw_binary_file: Input raw suite2p binary movie.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the binary conversion tool.
    :param int frames_include: [from suite2p docs] If greater than zero, only [the first] <frames_include> frames are processed. Useful for testing parameters on a subset of data.
    :param int align_by_chan: [from suite2p docs] Which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel). If you have a non-functional channel with something like td-Tomato expression, you may want to use this channel for alignment rather than the functional channel.
    :param int nimg_init: [from suite2p docs] How many frames to use to compute reference image for registration.
    :param int batch_size: [from suite2p docs] How many frames to register simultaneously in each batch. This depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.
    :param float maxregshift: [from suite2p docs] The maximum shift as a fraction of the frame size. If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops['maxregshift'].
    :param float smooth_sigma: [from suite2p docs] Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).
    :param float smooth_sigma_time: [from suite2p docs] Standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed. Might need this to be set to 1 or 2 for low SNR data.
    :param bool two_step_registration: [from suite2p docs] Whether or not to run registration twice (for low SNR data). `keep_movie_raw` must be True for this to work.
    :param int subpixel: [from suite2p docs] Precision of Subpixel Registration (1/subpixel steps).
    :param float th_badframes: [from suite2p docs] Involved with setting threshold for excluding frames for cropping. Set this smaller to exclude more frames.
    :param bool norm_frames: [from suite2p docs] Normalize frames when detecting shifts.
    :param bool force_refImg: [from suite2p docs] Specifies whether to use refImg stored in ops. Make sure that ops['refImg'] has a valid file pathname.
    :param bool pad_fft: [from suite2p docs] Specifies whether to pad image or not during FFT portion of registration.
    :param bool one_p_reg: [from suite2p docs] Whether to perform high-pass spatial filtering and tapering (parameters set below), which help with 1P registration.
    :param int spatial_hp_reg: [from suite2p docs] Window in pixels for spatial high-pass filtering before registration.
    :param float pre_smooth: [from suite2p docs] If > 0, defines stddev of Gaussian smoothing, which is applied before spatial high-pass filtering.
    :param float spatial_taper: [from suite2p docs] How many pixels to ignore on edges - they are set to zero (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma']).
    :param bool nonrigid: [from suite2p docs] Whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
    :param List[int] block_size: [from suite2p docs] Size of blocks for non-rigid registration, in pixels. HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient FFT.
    :param float snr_thresh: [from suite2p docs] How big the phase correlation peak has to be relative to the noise in the phase correlation map for the block shift to be accepted. In low SNR recordings like one-photon, I'd recommend a larger value like 1.5, so that block shifts are only accepted if there is significant SNR in the phase correlation.
    :param float maxregshiftNR: [from suite2p docs] Maximum shift in pixels of a block relative to the rigid shift.
    :param bool do_bidiphase: [from suite2p docs] Whether or not to compute bidirectional phase offset from misaligned line scanning experiment (applies to 2P recordings only). suite2p will estimate the bidirectional phase offset from ops['nimg_init'] frames if this is set to 1 (and ops['bidiphase']=0), and then apply this computed offset to all frames.
    :param int bidiphase: [from suite2p docs] Bidirectional phase offset from line scanning (set by user). If set to any value besides 0, then this offset is used and applied to all frames in the recording.
    :param bool bidi_corrected: [from suite2p docs] Specifies whether to do bidi correction.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param float viz_display_rate: Display rate for the preview movies, in Hz.
    """
    # define registered binary movie name
    ideas_output_dir = os.getcwd()
    reg_binary_path = f"{ideas_output_dir}/data.bin"
    ops_path = f"{ideas_output_dir}/ops_registration.npy"

    # temporarily copy the raw bin input file, since it's modified by suite2p during processing
    tmp_raw_binary_file = f"{ideas_output_dir}/tmp_data_raw.bin"
    shutil.copyfile(raw_binary_file[0], tmp_raw_binary_file)

    # load input parameter file
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)
    ops["raw_file"] = tmp_raw_binary_file
    ops["reg_file"] = reg_binary_path
    ops["ops_path"] = ops_path

    # set user-defined parameters
    ops["frames_include"] = frames_include
    ops["align_by_chan"] = align_by_chan
    ops["nimg_init"] = nimg_init
    ops["batch_size"] = batch_size
    ops["maxregshift"] = maxregshift
    ops["smooth_sigma"] = smooth_sigma
    ops["smooth_sigma_time"] = smooth_sigma_time
    ops["two_step_registration"] = two_step_registration
    ops["subpixel"] = subpixel
    ops["th_badframes"] = th_badframes
    ops["norm_frames"] = norm_frames
    ops["force_refImg"] = force_refImg
    ops["pad_fft"] = pad_fft
    ops["1Preg"] = one_p_reg
    ops["spatial_hp_reg"] = spatial_hp_reg
    ops["pre_smooth"] = pre_smooth
    ops["spatial_taper"] = spatial_taper
    ops["nonrigid"] = nonrigid
    ops["block_size"] = block_size
    ops["snr_thresh"] = snr_thresh
    ops["maxregshiftNR"] = maxregshiftNR
    ops["do_bidiphase"] = do_bidiphase
    ops["bidiphase"] = bidiphase
    ops["bidi_corrected"] = bidi_corrected

    Ly, Lx = ops["Ly"], ops["Lx"]

    # load input raw binary movie and create the output registered binary movie
    f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=ops["raw_file"], write=True)
    f_reg = io.BinaryFile(
        Ly=Ly, Lx=Lx, filename=ops["reg_file"], n_frames=f_raw.shape[0], write=True
    )  # Set registered binary file to have same n_frames

    # [start of suite2p code]
    # REGISTRATION
    t11 = time.time()
    plane_times = {}
    logger.info("----------- REGISTRATION")
    refImg = (
        ops["refImg"] if "refImg" in ops and ops.get("force_refImg", False) else None
    )

    align_by_chan2 = ops["functional_chan"] != ops["align_by_chan"]
    f_reg_chan2 = None
    registration_outputs = registration.registration_wrapper(
        f_reg,
        f_raw=f_raw,
        f_reg_chan2=f_reg_chan2,
        f_raw_chan2=None,
        refImg=refImg,
        align_by_chan2=align_by_chan2,
        ops=ops,
    )

    ops = registration.save_registration_outputs_to_ops(registration_outputs, ops)
    # add enhanced mean image
    meanImgE = registration.compute_enhanced_mean_image(
        ops["meanImg"].astype(np.float32), ops
    )
    ops["meanImgE"] = meanImgE
    # Inscopix edit: adding max projection image
    ops["max_proj"] = np.max(f_reg.data, axis=0)

    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)

    plane_times["registration"] = time.time() - t11
    logger.info("----------- Total %0.2f sec" % plane_times["registration"])
    n_frames, Ly, Lx = f_reg.shape

    if ops["two_step_registration"] and ops["keep_movie_raw"]:
        logger.info("----------- REGISTRATION STEP 2")
        logger.info("(making mean image (excluding bad frames)")
        nsamps = min(n_frames, 1000)
        inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
        if align_by_chan2:
            refImg = f_reg_chan2[inds].astype(np.float32).mean(axis=0)
        else:
            refImg = f_reg[inds].astype(np.float32).mean(axis=0)
        registration_outputs = registration.registration_wrapper(
            f_reg,
            f_raw=None,
            f_reg_chan2=f_reg_chan2,
            f_raw_chan2=None,
            refImg=refImg,
            align_by_chan2=align_by_chan2,
            ops=ops,
        )
        if ops.get("ops_path"):
            np.save(ops["ops_path"], ops)
        plane_times["two_step_registration"] = time.time() - t11
        logger.info("----------- Total %0.2f sec" % plane_times["two_step_registration"])

    # compute metrics for registration
    if ops.get("do_regmetrics", True) and n_frames >= 1500:
        t0 = time.time()
        # n frames to pick from full movie
        nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames)
        inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
        mov = f_reg[inds]
        mov = mov[
            :,
            ops["yrange"][0] : ops["yrange"][-1],
            ops["xrange"][0] : ops["xrange"][-1],
        ]
        ops = registration.get_pc_metrics(mov, ops)
        plane_times["registration_metrics"] = time.time() - t0
        logger.info("Registration metrics, %0.2f sec." % plane_times["registration_metrics"])
        if ops.get("ops_path"):
            np.save(ops["ops_path"], ops)
    # [end of suite2p code]

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="registration",
    )

    # output previews
    preview.create_output_previews(
        ops=ops,
        steps="registration",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
        display_rate=int(viz_display_rate),
    )

    # remove tmp files
    os.remove(tmp_raw_binary_file)

    logger.info("ALL DONE!")


def suite2p_roi_detection(
    *,
    reg_binary_file: List[str],
    ops_file: List[str],
    classifier_path: Optional[List[str]] = None,
    tau: float = 1.0,
    sparse_mode: bool = True,
    spatial_scale: int = 0,
    connected: bool = True,
    threshold_scaling: float = 1.0,
    spatial_hp_detect: int = 25,
    max_overlap: float = 0.75,
    high_pass: int = 100,
    smooth_masks: bool = True,
    max_iterations: int = 20,
    nbinned: int = 5000,
    denoise: bool = False,
    anatomical_only: int = 0,
    diameter: int = 0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 1.5,
    spatial_hp_cp: int = 0,
    pretrained_model: str = "cyto",
    preclassify: float = 0.0,
    chan2_thres: float = 0.65,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
):
    """
    Tool to run suite2p ROI detection on a registered suite2p binary movie. This constitutes the third step of the suite2p end-to-end pipeline.

    :param List[str] reg_binary_file: Input registered suite2p binary movie.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the registration tool.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param bool sparse_mode: [from suite2p docs] Whether or not to use sparse_mode cell detection.
    :param int spatial_scale: [from suite2p docs] What the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).
    :param bool connected: [from suite2p docs] Whether or not to require ROIs to be fully connected (set to 0 for dendrites/boutons).
    :param float threshold_scaling: [from suite2p docs] This controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected). if you set this higher, then fewer ROIs will be detected, and if you set it lower, more ROIs will be detected.
    :param int spatial_hp_detect: [from suite2p docs] Window for spatial high-pass filtering for neuropil subtracation before ROI detection takes place.
    :param float max_overlap: [from suite2p docs] We allow overlapping ROIs during cell detection. After detection, ROIs with more than ops['max_overlap'] fraction of their pixels overlapping with other ROIs will be discarded. Therefore, to throw out NO ROIs, set this to 1.0.
    :param int high_pass: [from suite2p docs] Running mean subtraction across time with window of size 'high_pass'. Values of less than 10 are recommended for 1P data where there are often large full-field changes in brightness.
    :param bool smooth_masks: [from suite2p docs] Whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
    :param int max_iterations: [from suite2p docs] How many iterations over which to extract cells - at most ops['max_iterations'], but usually stops before due to ops['threshold_scaling'] criterion.
    :param int nbinned: [from suite2p docs] Maximum number of binned frames to use for ROI detection.
    :param bool denoise: [from suite2p docs] Whether or not binned movie should be denoised before cell detection in sparse_mode. If True, make sure to set ops['sparse_mode'] is also set to True.
    :param int anatomical_only: [from suite2p docs] If greater than 0, specifies what to use Cellpose on.  1: Will find masks on max projection image divided by mean image.  2: Will find masks on mean image  3: Will find masks on enhanced mean image  4: Will find masks on maximum projection image.
    :param int diameter: [from suite2p docs] Diameter that will be used for cellpose. If set to zero, diameter is estimated.
    :param float cellprob_threshold: [from suite2p docs] Specifies threshold for cell detection that will be used by cellpose.
    :param float flow_threshold: [from suite2p docs] Specifies flow threshold that will be used for cellpose.
    :param int spatial_hp_cp: [from suite2p docs] Window for spatial high-pass filtering of image to be used for cellpose.
    :param str pretrained_model: [from suite2p docs] Path to pretrained model or string for model type (can be user's model ).
    :param float preclassify: [from suite2p docs] Apply classifier before signal extraction with probability threshold of 'preclassify'. If this is set to 0.0, then all detected ROIs are kept and signals are computed.
    :param float chan2_thres: [from suite2p docs] Threshold for calling an ROI "detected" on a second channel.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    """
    if isinstance(classifier_path, list) and len(classifier_path) == 0:
        logger.info(
            f"`classifier_path` input as {classifier_path}: replacing by `None`."
        )
        classifier_path = None

    # load input files
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)
    Ly, Lx = ops["Ly"], ops["Lx"]
    f_reg = io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_binary_file[0])
    n_frames, Ly, Lx = f_reg.shape

    # set user-defined parameters
    if classifier_path is not None:
        ops["classifier_path"] = classifier_path[0]
    ops["tau"] = tau
    ops["sparse_mode"] = sparse_mode
    ops["spatial_scale"] = spatial_scale
    ops["connected"] = connected
    ops["threshold_scaling"] = threshold_scaling
    ops["spatial_hp_detect"] = spatial_hp_detect
    ops["max_overlap"] = max_overlap
    ops["high_pass"] = high_pass
    ops["smooth_masks"] = smooth_masks
    ops["max_iterations"] = max_iterations
    ops["nbinned"] = nbinned
    ops["denoise"] = denoise
    ops["anatomical_only"] = anatomical_only
    ops["diameter"] = diameter
    ops["cellprob_threshold"] = cellprob_threshold
    ops["flow_threshold"] = flow_threshold
    ops["spatial_hp_cp"] = spatial_hp_cp
    ops["pretrained_model"] = pretrained_model
    ops["preclassify"] = preclassify
    ops["chan2_thres"] = chan2_thres

    # [start of suite2p code]
    # Select file for classification
    ops_classfile = ops.get("classifier_path")
    builtin_classfile = classification.builtin_classfile
    user_classfile = classification.user_classfile
    if ops_classfile:
        logger.info(f"NOTE: applying classifier {str(ops_classfile)}")
        classfile = ops_classfile
    elif ops["use_builtin_classifier"] or not user_classfile.is_file():
        logger.info(f"NOTE: Applying builtin classifier at {str(builtin_classfile)}")
        classfile = builtin_classfile
    else:
        logger.info(f"NOTE: applying default {str(user_classfile)}")
        classfile = user_classfile

    # CELL DETECTION
    t11 = time.time()
    plane_times = {}
    logger.info("----------- ROI DETECTION")
    ops, stat = detection.detection_wrapper(f_reg, ops=ops, classfile=classfile)
    plane_times["detection"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["detection"])
    # [end of suite2p code]

    # save output files into output folder
    ideas_output_dir = os.getcwd()
    ops["save_path"] = ideas_output_dir
    np.save(f"{ideas_output_dir}/stat_ROI_detection.npy", stat)
    np.save(f"{ideas_output_dir}/ops_ROI_detection.npy", ops)

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="roi_detection",
    )

    # output previews
    preview.create_output_previews(
        ops=ops,
        steps="roi_detection",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
    )

    logger.info("ALL DONE!")


def suite2p_roi_extraction(
    *,
    reg_binary_file: List[str],
    stat_file: List[str],
    ops_file: List[str],
    neuropil_extract: bool = True,
    allow_overlap: bool = False,
    min_neuropil_pixels: int = 350,
    inner_neuropil_radius: int = 2,
    lam_percentile: float = 50.0,
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Tool to run suite2p ROI extraction on a registered suite2p binary movie, using the previously detected ROIs. This constitutes the fourth step of the suite2p end-to-end pipeline.

    :param List[str] reg_binary_file: Input registered suite2p binary movie.
    :param List[str] stat_file: Input cell statistics file, as outputted by the ROI detection step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI detection tool.
    :param bool neuropil_extract: [from suite2p docs] Whether or not to extract signal from neuropil. If False, Fneu is set to zero.
    :param bool allow_overlap: [from suite2p docs] Whether or not to extract signals from pixels which belong to two ROIs. By default, any pixels which belong to two ROIs (overlapping pixels) are excluded from the computation of the ROI trace.
    :param int min_neuropil_pixels: [from suite2p docs] Minimum number of pixels used to compute neuropil for each cell.
    :param int inner_neuropil_radius: [from suite2p docs] Number of pixels to keep between ROI and neuropil donut.
    :param int lam_percentile: [from suite2p docs] Percentile of Lambda within area to ignore when excluding cell pixels for neuropil extraction.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    # load input files
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)
    Ly, Lx = ops["Ly"], ops["Lx"]
    f_reg = io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_binary_file[0])
    stat = np.load(stat_file[0], allow_pickle=True)

    # set user-defined parameters
    ops["neuropil_extract"] = neuropil_extract
    ops["allow_overlap"] = allow_overlap
    ops["min_neuropil_pixels"] = min_neuropil_pixels
    ops["inner_neuropil_radius"] = inner_neuropil_radius
    ops["lam_percentile"] = lam_percentile

    # [start of suite2p code]
    # ROI EXTRACTION
    t11 = time.time()
    plane_times = {}
    logger.info("----------- EXTRACTION")
    f_reg_chan2 = None
    stat, F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(
        stat, f_reg, f_reg_chan2=f_reg_chan2, ops=ops
    )

    plane_times["extraction"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["extraction"])
    # [end of suite2p code]

    # save output files into output folder
    ideas_output_dir = os.getcwd()
    ops["save_path"] = ideas_output_dir
    np.save(f"{ideas_output_dir}/stat.npy", stat)
    np.save(f"{ideas_output_dir}/F.npy", F)
    np.save(f"{ideas_output_dir}/Fneu.npy", Fneu)
    np.save(f"{ideas_output_dir}/ops_ROI_extraction.npy", ops)

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="roi_extraction",
    )

    # output previews
    preview.create_output_previews(
        ops=ops,
        steps="roi_extraction",
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
        n_samp_cells=int(viz_n_samp_cells),
        random_seed=int(viz_random_seed),
        show_all_footprints=viz_show_all_footprints,
    )

    logger.info("ALL DONE!")


def suite2p_roi_classification(
    *,
    stat_file: List[str],
    ops_file: List[str],
    classifier_path: Optional[List[str]] = None,
    soma_crop: bool = True,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
):
    """
    Tool to run suite2p ROI classification on the extracted ROIs. This constitutes the fifth step of the suite2p end-to-end pipeline.

    :param List[str] stat_file: Input cell statistics file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI extraction tool.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param bool soma_crop: [from suite2p docs] Specifies whether to crop dendrites for cell classification stats (e.g., compactness).
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    """
    # load input files
    stat = np.load(stat_file[0], allow_pickle=True)
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)

    # set user-defined parameters
    if classifier_path is not None:
        ops["classifier_path"] = classifier_path[0]
    ops["soma_crop"] = soma_crop

    # [start of suite2p code]
    # Select file for classification
    ops_classfile = ops.get("classifier_path")
    builtin_classfile = classification.builtin_classfile
    user_classfile = classification.user_classfile
    if ops_classfile:
        logger.info(f"NOTE: applying classifier {str(ops_classfile)}")
        classfile = ops_classfile
    elif ops["use_builtin_classifier"] or not user_classfile.is_file():
        logger.info(f"NOTE: Applying builtin classifier at {str(builtin_classfile)}")
        classfile = builtin_classfile
    else:
        logger.info(f"NOTE: applying default {str(user_classfile)}")
        classfile = user_classfile

    # ROI CLASSIFICATION
    t11 = time.time()
    plane_times = {}
    logger.info("----------- CLASSIFICATION")
    if len(stat):
        iscell = classification.classify(stat=stat, classfile=classfile)
    else:
        iscell = np.zeros((0, 2))
    plane_times["classification"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["classification"])
    # [end of suite2p code]

    # save output file into output folder
    ideas_output_dir = os.getcwd()
    ops["save_path"] = ideas_output_dir
    np.save(f"{ideas_output_dir}/iscell.npy", iscell)
    np.save(f"{ideas_output_dir}/ops_ROI_classification.npy", ops)

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="roi_classification",
    )

    # output previews
    # temporarily copy input stat file to output dir for preview generation
    tmp_stat_file = None
    if not os.path.exists(f"{ideas_output_dir}/stat.npy"):
        tmp_stat_file = f"{ideas_output_dir}/stat.npy"
        shutil.copy(stat_file[0], tmp_stat_file)

    preview.create_output_previews(
        ops=ops,
        steps="roi_classification",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
    )

    if tmp_stat_file:
        os.remove(tmp_stat_file)

    logger.info("ALL DONE!")


def suite2p_spike_deconvolution(
    *,
    fluo_file: List[str],
    neuropil_fluo_file: List[str],
    ops_file: List[str],
    tau: float = 1.0,
    neucoeff: float = 0.7,
    baseline: str = "maximin",
    win_baseline: float = 60.0,
    sig_baseline: float = 10.0,
    prctile_baseline: float = 8.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Tool to run suite2p spike deconvolution on the extracted fluorescence traces. This constitutes the sixth step of the suite2p end-to-end pipeline.

    :param List[str] fluo_file: Input fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] neuropil_fluo_file: Input neuropil fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI extraction tool.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param float neucoeff: [from suite2p docs] Neuropil coefficient for all ROIs.
    :param str baseline: [from suite2p docs] How to compute the baseline of each trace. This baseline is then subtracted from each cell. 'maximin' computes a moving baseline by filtering the data with a Gaussian of width ops['sig_baseline'] * ops['fs'], and then minimum filtering with a window of ops['win_baseline'] * ops['fs'], and then maximum filtering with the same window. 'constant' computes a constant baseline by filtering with a Gaussian of width ops['sig_baseline'] * ops['fs'] and then taking the minimum value of this filtered trace. 'constant_percentile' computes a constant baseline by taking the ops['prctile_baseline'] percentile of the trace.
    :param float win_baseline: [from suite2p docs] Window for maximin filter in seconds.
    :param float sig_baseline: [from suite2p docs] Gaussian filter width in seconds, used before maximin filtering or taking the minimum value of the trace, ops['baseline'] = 'maximin' or 'constant'.
    :param float prctile_baseline: [from suite2p docs] Percentile of trace to use as baseline if ops['baseline'] = 'constant_percentile'.
    :param int viz_n_samp_cells: Number of sample cells for the spike deconvolution preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    # load input files
    F = np.load(fluo_file[0], allow_pickle=True)
    Fneu = np.load(neuropil_fluo_file[0], allow_pickle=True)
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)

    # set user-defined parameters
    ops["tau"] = tau
    ops["neucoeff"] = neucoeff
    ops["baseline"] = baseline
    ops["win_baseline"] = win_baseline
    ops["sig_baseline"] = sig_baseline
    ops["prctile_baseline"] = prctile_baseline

    # [start of suite2p code]
    # SPIKE DECONVOLUTION
    t11 = time.time()
    plane_times = {}
    logger.info("----------- SPIKE DECONVOLUTION")
    dF = F.copy() - ops["neucoeff"] * Fneu
    dF = extraction.preprocess(
        F=dF,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
    )
    spks = extraction.oasis(
        F=dF, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"]
    )
    plane_times["deconvolution"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["deconvolution"])
    # [end of suite2p code]

    # save output file into output folder
    ideas_output_dir = os.getcwd()
    ops["save_path"] = ideas_output_dir
    np.save(f"{ideas_output_dir}/spks.npy", spks)
    np.save(f"{ideas_output_dir}/ops_spike_deconvolution.npy", ops)

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="spike_deconvolution",
    )

    # output previews
    # temporarily copy input fluo file to output dir for preview generation
    tmp_fluo_file, tmp_neuropli_file = None, None
    if not os.path.exists(f"{ideas_output_dir}/F.npy"):
        tmp_fluo_file = f"{ideas_output_dir}/F.npy"
        shutil.copy(fluo_file[0], tmp_fluo_file)
    if not os.path.exists(f"{ideas_output_dir}/Fneu.npy"):
        tmp_neuropli_file = f"{ideas_output_dir}/Fneu.npy"
        shutil.copy(neuropil_fluo_file[0], tmp_neuropli_file)

    preview.create_output_previews(
        ops=ops,
        steps="spike_deconvolution",
        n_samp_cells=int(viz_n_samp_cells),
        random_seed=int(viz_random_seed),
        show_all_footprints=viz_show_all_footprints,
    )

    if tmp_fluo_file:
        os.remove(tmp_fluo_file)
    if tmp_neuropli_file:
        os.remove(tmp_neuropli_file)

    logger.info("ALL DONE!")


def suite2p_output_conversion(
    *,
    fluo_file: List[str],
    neuropil_fluo_file: List[str],
    spks_file: List[str],
    stat_file: List[str],
    ops_file: List[str],
    iscell_file: List[str],
    save_npy: bool = True,
    save_isxd: bool = True,
    save_NWB: bool = False,
    save_mat: bool = False,
    thresh_spks_perc: float = 99.7,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Tool to output suite2p results in specific formats. This constitutes the seventh and last step of the suite2p end-to-end pipeline.

    :param List[str] fluo_file: Input fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] neuropil_fluo_file: Input neuropil fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] spks_file: Input deconvolved spikes file, as outputted by the spike deconvolution step.
    :param List[str] stat_file: Input extraction statistics file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input parameters file, as outputted by the spike deconvolution step.
    :param List[str] iscell_file: Input classification labels file, as outputted by the ROI classification step.
    :param bool save_npy: If true, save suite2p NPY output (as a ZIP file).
    :param bool save_isxd: If true, save suite2p output as ISXD files.
    :param bool save_NWB: [from suite2p docs] Whether to save output as NWB file.
    :param bool save_mat: [from suite2p docs] Whether to save the results in matlab format in file "Fall.mat".
    :param float thresh_spks_perc: Threshold for denoising the deconvolved spike trains, in percentile. Any value in the ROIs-by-time-point deconvolved spike matrix that is below the matrix's xth percentile value is set to 0. Note that the same threshold is applied to all ROIs, and that this thresholding step does not binarize the deconvolved spike trains but simply filters out the low-amplitude spike events.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    # load input files
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)
    file_list = [
        fluo_file,
        neuropil_fluo_file,
        spks_file,
        stat_file,
        ops_file,
        iscell_file,
    ]
    suite2p_output_dir = os.path.dirname(fluo_file[0])
    ideas_output_dir = os.getcwd()

    # set user-defined parameters
    ops["save_npy"] = save_npy
    ops["save_isxd"] = save_isxd
    ops["save_NWB"] = save_NWB
    ops["save_mat"] = save_mat
    ops["save_path"] = suite2p_output_dir

    # output metadata
    metadata.create_output_metadata(
        ops=ops,
        steps="output_conversion",
    )

    # output preview(s)
    preview.create_output_previews(
        ops=ops,
        steps="output_conversion",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
        n_samp_cells=int(viz_n_samp_cells),
        thresh_spks_perc=thresh_spks_perc,
        random_seed=int(viz_random_seed),
        show_all_footprints=viz_show_all_footprints,
    )

    if save_npy:
        # zip suite2p outputs
        zip_output_file = f"{ideas_output_dir}/suite2p_output.zip"
        with ZipFile(zip_output_file, "w") as f:
            for file in file_list:
                f.write(file[0])
    if save_isxd:
        fname_list = [os.path.basename(x[0]) for x in file_list]
        del fname_list[1]
        tlbxio.npy_to_isxd(
            npy_dir=suite2p_output_dir,
            output_dir=ideas_output_dir,
            thresh_spks_perc=thresh_spks_perc,
            custom_fnames=fname_list,
        )
    if save_NWB:
        # suite2p.io.nwb.save_nwb() looks for a "plane*" directory to locate an "ops.npy" file
        plane_output_dir = os.path.join(ideas_output_dir, "plane0")
        if not os.path.exists(plane_output_dir):
            os.mkdir(plane_output_dir)
        fixed_fname_list = [
            f"{x}.npy" for x in ["F", "Fneu", "spks", "stat", "ops", "iscell"]
        ]
        for file, fixed_fname in zip(file_list, fixed_fname_list):
            shutil.copyfile(file[0], os.path.join(plane_output_dir, fixed_fname))
        ops["save_path"] = plane_output_dir
        np.save(os.path.join(plane_output_dir, "ops.npy"), ops)
        io.save_nwb(ideas_output_dir)
        shutil.rmtree(plane_output_dir)
    if save_mat:
        # suite2p.io.save.save_mat() calls scipy.io.savemat() with file_name based on ops["save_path"]
        F = np.load(fluo_file[0], allow_pickle=True)
        Fneu = np.load(neuropil_fluo_file[0], allow_pickle=True)
        spks = np.load(spks_file[0], allow_pickle=True)
        stat = np.load(stat_file[0], allow_pickle=True)
        iscell = np.load(iscell_file[0], allow_pickle=True)
        ops["save_path"] = os.path.dirname(fluo_file[0])
        io.save_mat(
            ops,
            stat,
            F,
            Fneu,
            spks,
            iscell,
            redcell=[],
            F_chan2=None,
            Fneu_chan2=None,
        )

    # output metadata
    metadata.create_output_metadata(ops=ops, steps="output_conversion")


# ================================ IDEAS Wrapper Functions ====================================


def suite2p_binary_conversion_ideas_wrapper(
    *,
    raw_movie_files: List[IdeasFile],
    nplanes: int = 1,
    nchannels: int = 1,
    functional_chan: int = 1,
    fs: Optional[float] = None,
    bruker_bidirectional: bool = False,
):
    """
    Ideas wrapper for tool to convert raw 2P input movie(s) into a suite2p binary file. This constitutes the first step of the suite2p end-to-end pipeline.

    :param List[str] raw_movie_files: Input 2P movie(s) [.isxd, .zip, .tiff].
    :param int nplanes: [from suite2p docs] Each tiff has this many planes in sequence.
    :param int nchannels: [from suite2p docs] Each tiff has this many channels per plane.
    :param int functional_chan: [from suite2p docs] This channel is used to extract functional ROIs (1-based, so 1 means first channel, and 2 means second channel).
    :param Optional[float] fs: [from suite2p docs] Sampling rate (per plane). For instance, if you have a 10 plane recording acquired at 30Hz, then the sampling rate per plane is 3Hz, so set ops['fs'] = 3.
    :param bool bruker_bidirectional: [from suite2p docs] Specifies whether BRUKER files are bidirectional multiplane recordings. The True setting corresponds to the following plane order (first plane is indexed as zero): [0,1,2,2,1,0]. False corresponds to [0,1,2,0,1,2].
    """

    suite2p_binary_conversion(
        raw_movie_files=raw_movie_files,
        nplanes=nplanes,
        nchannels=nchannels,
        functional_chan=functional_chan,
        fs=fs,
        bruker_bidirectional=bruker_bidirectional,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        output_prefix = outputs.input_paths_to_output_prefix(
            raw_movie_files, max_name_len=100
        )
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "data_raw.bin",
                prefix=output_prefix
            ).register_preview(
                "movie_preview.mp4",
                caption="Preview raw binary movie (from data_raw.bin)"
            ).register_metadata_dict(
                **metadata["data_raw"]
            )
            output_data.register_file(
                "ops_binary_conversion.npy",
                prefix=output_prefix,
            )
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_registration_ideas_wrapper(
    *,
    raw_binary_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    frames_include: int = -1,
    align_by_chan: int = 1,
    nimg_init: int = 300,
    batch_size: int = 500,
    maxregshift: float = 0.1,
    smooth_sigma: float = 1.15,
    smooth_sigma_time: float = 0.0,
    two_step_registration: bool = False,
    subpixel: int = 10,
    th_badframes: float = 1.0,
    norm_frames: bool = True,
    force_refImg: bool = False,
    pad_fft: bool = False,
    one_p_reg: bool = False,
    spatial_hp_reg: int = 42,
    pre_smooth: float = 0.0,
    spatial_taper: float = 40.0,
    nonrigid: bool = True,
    block_size: List[int] = [128, 128],
    snr_thresh: float = 1.2,
    maxregshiftNR: float = 5.0,
    do_bidiphase: bool = False,
    bidiphase: int = 0,
    bidi_corrected: bool = False,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_display_rate: float = 10.0,
):
    """
    Ideas wrapper for tool to run suite2p registration on a raw suite2p binary movie. This constitutes the second step of the suite2p end-to-end pipeline.

    :param List[str] raw_binary_file: Input raw suite2p binary movie.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the binary conversion tool.
    :param int frames_include: [from suite2p docs] If greater than zero, only [the first] <frames_include> frames are processed. Useful for testing parameters on a subset of data.
    :param int align_by_chan: [from suite2p docs] Which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel). If you have a non-functional channel with something like td-Tomato expression, you may want to use this channel for alignment rather than the functional channel.
    :param int nimg_init: [from suite2p docs] How many frames to use to compute reference image for registration.
    :param int batch_size: [from suite2p docs] How many frames to register simultaneously in each batch. This depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.
    :param float maxregshift: [from suite2p docs] The maximum shift as a fraction of the frame size. If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops['maxregshift'].
    :param float smooth_sigma: [from suite2p docs] Standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).
    :param float smooth_sigma_time: [from suite2p docs] Standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed. Might need this to be set to 1 or 2 for low SNR data.
    :param bool two_step_registration: [from suite2p docs] Whether or not to run registration twice (for low SNR data). `keep_movie_raw` must be True for this to work.
    :param int subpixel: [from suite2p docs] Precision of Subpixel Registration (1/subpixel steps).
    :param float th_badframes: [from suite2p docs] Involved with setting threshold for excluding frames for cropping. Set this smaller to exclude more frames.
    :param bool norm_frames: [from suite2p docs] Normalize frames when detecting shifts.
    :param bool force_refImg: [from suite2p docs] Specifies whether to use refImg stored in ops. Make sure that ops['refImg'] has a valid file pathname.
    :param bool pad_fft: [from suite2p docs] Specifies whether to pad image or not during FFT portion of registration.
    :param bool one_p_reg: [from suite2p docs] Whether to perform high-pass spatial filtering and tapering (parameters set below), which help with 1P registration.
    :param int spatial_hp_reg: [from suite2p docs] Window in pixels for spatial high-pass filtering before registration.
    :param float pre_smooth: [from suite2p docs] If > 0, defines stddev of Gaussian smoothing, which is applied before spatial high-pass filtering.
    :param float spatial_taper: [from suite2p docs] How many pixels to ignore on edges - they are set to zero (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma']).
    :param bool nonrigid: [from suite2p docs] Whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
    :param List[int] block_size: [from suite2p docs] Size of blocks for non-rigid registration, in pixels. HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient FFT.
    :param float snr_thresh: [from suite2p docs] How big the phase correlation peak has to be relative to the noise in the phase correlation map for the block shift to be accepted. In low SNR recordings like one-photon, I'd recommend a larger value like 1.5, so that block shifts are only accepted if there is significant SNR in the phase correlation.
    :param float maxregshiftNR: [from suite2p docs] Maximum shift in pixels of a block relative to the rigid shift.
    :param bool do_bidiphase: [from suite2p docs] Whether or not to compute bidirectional phase offset from misaligned line scanning experiment (applies to 2P recordings only). suite2p will estimate the bidirectional phase offset from ops['nimg_init'] frames if this is set to 1 (and ops['bidiphase']=0), and then apply this computed offset to all frames.
    :param int bidiphase: [from suite2p docs] Bidirectional phase offset from line scanning (set by user). If set to any value besides 0, then this offset is used and applied to all frames in the recording.
    :param bool bidi_corrected: [from suite2p docs] Specifies whether to do bidi correction.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param float viz_display_rate: Display rate for the preview movies, in Hz.
    """

    suite2p_registration(
        raw_binary_file=raw_binary_file,
        ops_file=ops_file,
        frames_include=frames_include,
        align_by_chan=align_by_chan,
        nimg_init=nimg_init,
        batch_size=batch_size,
        maxregshift=maxregshift,
        smooth_sigma=smooth_sigma,
        smooth_sigma_time=smooth_sigma_time,
        two_step_registration=two_step_registration,
        subpixel=subpixel,
        th_badframes=th_badframes,
        norm_frames=norm_frames,
        force_refImg=force_refImg,
        pad_fft=pad_fft,
        one_p_reg=one_p_reg,
        spatial_hp_reg=spatial_hp_reg,
        pre_smooth=pre_smooth,
        spatial_taper=spatial_taper,
        nonrigid=nonrigid,
        block_size=block_size,
        snr_thresh=snr_thresh,
        maxregshiftNR=maxregshiftNR,
        do_bidiphase=do_bidiphase,
        bidiphase=bidiphase,
        bidi_corrected=bidi_corrected,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
        viz_display_rate=viz_display_rate,
    )
    
    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        output_prefix = outputs.input_paths_to_output_prefix(
            raw_binary_file, max_name_len=100
        )
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "data.bin",
                prefix=output_prefix,
            ).register_preview(
                "registration_fovs.svg",
                caption="Various FOVs from the registration process (from ops_registration.npy)"
            ).register_preview(
                "registration_offsets.svg",
                caption="x and y offsets for both rigid and non-rigid registration (from ops_registration.npy)"
            ).register_preview(
                "registration_movies.mp4",
                caption="Side-by-side raw and registered movies (from ops_registration.npy, data_raw.bin, and data.bin)"
            ).register_preview(
                "movie_preview.mp4",
                caption="Preview registered binary movie (from data.bin)"
            ).register_metadata_dict(
                **metadata["data"]
            )
            output_data.register_file(
                "ops_registration.npy",
                prefix=output_prefix,
            )
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_roi_detection_ideas_wrapper(
    *,
    reg_binary_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    classifier_path: Optional[List[IdeasFile]] = None,
    tau: float = 1.0,
    sparse_mode: bool = True,
    spatial_scale: int = 0,
    connected: bool = True,
    threshold_scaling: float = 1.0,
    spatial_hp_detect: int = 25,
    max_overlap: float = 0.75,
    high_pass: int = 100,
    smooth_masks: bool = True,
    max_iterations: int = 20,
    nbinned: int = 5000,
    denoise: bool = False,
    anatomical_only: int = 0,
    diameter: int = 0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 1.5,
    spatial_hp_cp: int = 0,
    pretrained_model: str = "cyto",
    preclassify: float = 0.0,
    chan2_thres: float = 0.65,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
):
    """
    Ideas wrapper for tool to run suite2p ROI detection on a registered suite2p binary movie. This constitutes the third step of the suite2p end-to-end pipeline.

    :param List[str] reg_binary_file: Input registered suite2p binary movie.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the registration tool.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param bool sparse_mode: [from suite2p docs] Whether or not to use sparse_mode cell detection.
    :param int spatial_scale: [from suite2p docs] What the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).
    :param bool connected: [from suite2p docs] Whether or not to require ROIs to be fully connected (set to 0 for dendrites/boutons).
    :param float threshold_scaling: [from suite2p docs] This controls the threshold at which to detect ROIs (how much the ROIs have to stand out from the noise to be detected). if you set this higher, then fewer ROIs will be detected, and if you set it lower, more ROIs will be detected.
    :param int spatial_hp_detect: [from suite2p docs] Window for spatial high-pass filtering for neuropil subtracation before ROI detection takes place.
    :param float max_overlap: [from suite2p docs] We allow overlapping ROIs during cell detection. After detection, ROIs with more than ops['max_overlap'] fraction of their pixels overlapping with other ROIs will be discarded. Therefore, to throw out NO ROIs, set this to 1.0.
    :param int high_pass: [from suite2p docs] Running mean subtraction across time with window of size 'high_pass'. Values of less than 10 are recommended for 1P data where there are often large full-field changes in brightness.
    :param bool smooth_masks: [from suite2p docs] Whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
    :param int max_iterations: [from suite2p docs] How many iterations over which to extract cells - at most ops['max_iterations'], but usually stops before due to ops['threshold_scaling'] criterion.
    :param int nbinned: [from suite2p docs] Maximum number of binned frames to use for ROI detection.
    :param bool denoise: [from suite2p docs] Whether or not binned movie should be denoised before cell detection in sparse_mode. If True, make sure to set ops['sparse_mode'] is also set to True.
    :param int anatomical_only: [from suite2p docs] If greater than 0, specifies what to use Cellpose on.  1: Will find masks on max projection image divided by mean image.  2: Will find masks on mean image  3: Will find masks on enhanced mean image  4: Will find masks on maximum projection image.
    :param int diameter: [from suite2p docs] Diameter that will be used for cellpose. If set to zero, diameter is estimated.
    :param float cellprob_threshold: [from suite2p docs] Specifies threshold for cell detection that will be used by cellpose.
    :param float flow_threshold: [from suite2p docs] Specifies flow threshold that will be used for cellpose.
    :param int spatial_hp_cp: [from suite2p docs] Window for spatial high-pass filtering of image to be used for cellpose.
    :param str pretrained_model: [from suite2p docs] Path to pretrained model or string for model type (can be user's model ).
    :param float preclassify: [from suite2p docs] Apply classifier before signal extraction with probability threshold of 'preclassify'. If this is set to 0.0, then all detected ROIs are kept and signals are computed.
    :param float chan2_thres: [from suite2p docs] Threshold for calling an ROI "detected" on a second channel.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    """

    suite2p_roi_detection(
        reg_binary_file=reg_binary_file,
        ops_file=ops_file,
        classifier_path=classifier_path,
        tau=tau,
        sparse_mode=sparse_mode,
        spatial_scale=spatial_scale,
        connected=connected,
        threshold_scaling=threshold_scaling,
        spatial_hp_detect=spatial_hp_detect,
        max_overlap=max_overlap,
        high_pass=high_pass,
        smooth_masks=smooth_masks,
        max_iterations=max_iterations,
        nbinned=nbinned,
        denoise=denoise,
        anatomical_only=anatomical_only,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        spatial_hp_cp=spatial_hp_cp,
        pretrained_model=pretrained_model,
        preclassify=preclassify,
        chan2_thres=chan2_thres,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        output_prefix = outputs.input_paths_to_output_prefix(
            reg_binary_file, max_name_len=100
        )
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "stat_ROI_detection.npy",
                prefix=output_prefix,
            ).register_preview(
                "detection_footprints_all.svg",
                caption="Various FOVs from the ROI detection process (from ops.npy and stat_ROI_detection.npy)"
            ).register_preview(
                "detection_footprints_detected.svg",
                caption="FOV of the detected ROIs (from ops.npy and stat_ROI_detection.npy)"
            ).register_metadata_dict(
                **metadata["stat_ROI_detection"]
            )
            output_data.register_file(
                "ops_ROI_detection.npy",
                prefix=output_prefix,
            )
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_roi_extraction_ideas_wrapper(
    *,
    reg_binary_file: List[IdeasFile],
    stat_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    neuropil_extract: bool = True,
    allow_overlap: bool = False,
    min_neuropil_pixels: int = 350,
    inner_neuropil_radius: int = 2,
    lam_percentile: float = 50.0,
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Ideas wrapper for tool to run suite2p ROI extraction on a registered suite2p binary movie, using the previously detected ROIs. This constitutes the fourth step of the suite2p end-to-end pipeline.

    :param List[str] reg_binary_file: Input registered suite2p binary movie.
    :param List[str] stat_file: Input cell statistics file, as outputted by the ROI detection step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI detection tool.
    :param bool neuropil_extract: [from suite2p docs] Whether or not to extract signal from neuropil. If False, Fneu is set to zero.
    :param bool allow_overlap: [from suite2p docs] Whether or not to extract signals from pixels which belong to two ROIs. By default, any pixels which belong to two ROIs (overlapping pixels) are excluded from the computation of the ROI trace.
    :param int min_neuropil_pixels: [from suite2p docs] Minimum number of pixels used to compute neuropil for each cell.
    :param int inner_neuropil_radius: [from suite2p docs] Number of pixels to keep between ROI and neuropil donut.
    :param int lam_percentile: [from suite2p docs] Percentile of Lambda within area to ignore when excluding cell pixels for neuropil extraction.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    suite2p_roi_extraction(
        reg_binary_file=reg_binary_file,
        stat_file=stat_file,
        ops_file=ops_file,
        neuropil_extract=neuropil_extract,
        allow_overlap=allow_overlap,
        min_neuropil_pixels=min_neuropil_pixels,
        inner_neuropil_radius=inner_neuropil_radius,
        lam_percentile=lam_percentile,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
        viz_n_samp_cells=viz_n_samp_cells,
        viz_random_seed=viz_random_seed,
        viz_show_all_footprints=viz_show_all_footprints,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        output_prefix = outputs.input_paths_to_output_prefix(
            reg_binary_file, stat_file, max_name_len=100
        )
        with outputs.register(raise_missing_file=False) as output_data:
            stat_file = output_data.register_file(
                "stat.npy",
                prefix=output_prefix
            ).register_metadata_dict(
                **metadata["stat"]
            )

            f_file = output_data.register_file(
                "F.npy",
                prefix=output_prefix,
            ).register_preview(
                "extracted_sample_sources_traces_only.svg",
                caption="Sample fluorescence traces only (from ops.npy and F.npy)",
            )
            
            for f in [stat_file, f_file]:
                f.register_preview(
                    "extracted_sample_sources_footprints.svg",
                    caption="Footprints of the sample sources (from ops.npy and stat.npy)",
                    prefix=""
                )

            output_data.register_file(
                "Fneu.npy",
                prefix=output_prefix,
            )

            output_data.register_file(
                "ops_ROI_extraction.npy",
                prefix=output_prefix,
            )
        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_roi_classification_ideas_wrapper(
    *,
    stat_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    classifier_path: Optional[List[IdeasFile]] = None,
    soma_crop: bool = True,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
):
    """
    Ideas wrapper for tool to run suite2p ROI classification on the extracted ROIs. This constitutes the fifth step of the suite2p end-to-end pipeline.

    :param List[str] stat_file: Input cell statistics file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI extraction tool.
    :param Optional[List[str]] classifier_path: [from suite2p docs] Path to classifier file you want to use for cell classification.
    :param bool soma_crop: [from suite2p docs] Specifies whether to crop dendrites for cell classification stats (e.g., compactness).
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    """
    suite2p_roi_classification(
        stat_file=stat_file,
        ops_file=ops_file,
        classifier_path=classifier_path,
        soma_crop=soma_crop,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
    )

    try:
        logger.info("Registering output data")
        metadata = outputs._load_and_remove_output_metadata()
        output_prefix = outputs.input_paths_to_output_prefix(
            stat_file, max_name_len=100
        )
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "iscell.npy",
                prefix=output_prefix
            ).register_preview(
                "detection_footprints_all.svg",
                caption="Various FOVs from the ROI detection process (from ops.npy, stat.npy, and iscell.npy)"
            ).register_preview(
                "detection_footprints_accepted.svg",
                caption="FOV of the accepted ROIs from the ROI detection process (from ops.npy, stat.npy, and iscell.npy)"
            ).register_metadata_dict(
                **metadata["iscell"]
            )
            output_data.register_file(
                "ops_ROI_classification.npy",
                prefix=output_prefix,
            )

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_spike_deconvolution_ideas_wrapper(
    *,
    fluo_file: List[IdeasFile],
    neuropil_fluo_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    tau: float = 1.0,
    neucoeff: float = 0.7,
    baseline: str = "maximin",
    win_baseline: float = 60.0,
    sig_baseline: float = 10.0,
    prctile_baseline: float = 8.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Tool to run suite2p spike deconvolution on the extracted fluorescence traces. This constitutes the sixth step of the suite2p end-to-end pipeline.

    :param List[str] fluo_file: Input fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] neuropil_fluo_file: Input neuropil fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input suite2p parameters file, as outputted by the ROI extraction tool.
    :param float tau: [from suite2p docs] The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f; 1.0 for GCaMP6m; 1.25-1.5 for GCaMP6s.
    :param float neucoeff: [from suite2p docs] Neuropil coefficient for all ROIs.
    :param str baseline: [from suite2p docs] How to compute the baseline of each trace. This baseline is then subtracted from each cell. 'maximin' computes a moving baseline by filtering the data with a Gaussian of width ops['sig_baseline'] * ops['fs'], and then minimum filtering with a window of ops['win_baseline'] * ops['fs'], and then maximum filtering with the same window. 'constant' computes a constant baseline by filtering with a Gaussian of width ops['sig_baseline'] * ops['fs'] and then taking the minimum value of this filtered trace. 'constant_percentile' computes a constant baseline by taking the ops['prctile_baseline'] percentile of the trace.
    :param float win_baseline: [from suite2p docs] Window for maximin filter in seconds.
    :param float sig_baseline: [from suite2p docs] Gaussian filter width in seconds, used before maximin filtering or taking the minimum value of the trace, ops['baseline'] = 'maximin' or 'constant'.
    :param float prctile_baseline: [from suite2p docs] Percentile of trace to use as baseline if ops['baseline'] = 'constant_percentile'.
    :param int viz_n_samp_cells: Number of sample cells for the spike deconvolution preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    suite2p_spike_deconvolution(
        fluo_file=fluo_file,
        neuropil_fluo_file=neuropil_fluo_file,
        ops_file=ops_file,
        tau=tau,
        neucoeff=neucoeff,
        baseline=baseline,
        win_baseline=win_baseline,
        sig_baseline=sig_baseline,
        prctile_baseline=prctile_baseline,
        viz_n_samp_cells=viz_n_samp_cells,
        viz_random_seed=viz_random_seed,
        viz_show_all_footprints=viz_show_all_footprints,
    )

    try:
        logger.info("Registering output data")
        output_prefix = outputs.input_paths_to_output_prefix(
            fluo_file, neuropil_fluo_file, ops_file, max_name_len=100
        )
        metadata = outputs._load_and_remove_output_metadata()
        with outputs.register(raise_missing_file=False) as output_data:
            output_data.register_file(
                "spks.npy",
                prefix=output_prefix
            ).register_preview(
                "extracted_sample_sources_traces_spikes.svg",
                caption="Sample fluorescence traces and deconvolved spikes (from ops.npy, F.npy, and spks.npy)"
            ).register_preview(
                "raster_deconvolved_spikes.svg",
                caption="Raster plot of the deconvolved spikes (from ops.npy, spks.npy, and iscell.npy)"
            ).register_preview(
                "extracted_sample_sources_traces_only.svg",
                caption="Sample fluorescence traces only (from ops.npy, F.npy, and iscell.npy)"
            ).register_preview(
                "extracted_sample_sources_traces_only.svg",
                caption="Sample fluorescence traces, neuropil traces and deconvolved spikes (from ops.npy, F.npy, Fneu.npy, spks.npy, and iscell.npy)"
            ).register_preview(
                "extracted_sample_sources_traces_suite2p.svg",
                caption="Sample fluorescence traces, neuropil traces and deconvolved spikes (from ops.npy, F.npy, Fneu.npy, spks.npy, and iscell.npy)"
            ).register_metadata_dict(
                **metadata["spks"]
            )

            output_data.register_file(
                "ops_spike_deconvolution.npy",
                prefix=output_prefix,
            )

        logger.info("Registered output data")
    except Exception:
        logger.exception("Failed to generate output data!")


def suite2p_output_conversion_ideas_wrapper(
    *,
    fluo_file: List[IdeasFile],
    neuropil_fluo_file: List[IdeasFile],
    spks_file: List[IdeasFile],
    stat_file: List[IdeasFile],
    ops_file: List[IdeasFile],
    iscell_file: List[IdeasFile],
    save_npy: bool = True,
    save_isxd: bool = True,
    save_NWB: bool = False,
    save_mat: bool = False,
    thresh_spks_perc: float = 99.7,
    viz_vmin_perc: float = 0.0,
    viz_vmax_perc: float = 99.0,
    viz_cmap: str = "plasma",
    viz_show_grid: bool = True,
    viz_ticks_step: float = 128.0,
    viz_n_samp_cells: int = 20,
    viz_random_seed: int = 0,
    viz_show_all_footprints: bool = True,
):
    """
    Ideas wrapper for tool to output suite2p results in specific formats. This constitutes the seventh and last step of the suite2p end-to-end pipeline.

    :param List[str] fluo_file: Input fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] neuropil_fluo_file: Input neuropil fluorescence traces file, as outputted by the ROI extraction step.
    :param List[str] spks_file: Input deconvolved spikes file, as outputted by the spike deconvolution step.
    :param List[str] stat_file: Input extraction statistics file, as outputted by the ROI extraction step.
    :param List[str] ops_file: Input parameters file, as outputted by the spike deconvolution step.
    :param List[str] iscell_file: Input classification labels file, as outputted by the ROI classification step.
    :param bool save_npy: If true, save suite2p NPY output (as a ZIP file).
    :param bool save_isxd: If true, save suite2p output as ISXD files.
    :param bool save_NWB: [from suite2p docs] Whether to save output as NWB file.
    :param bool save_mat: [from suite2p docs] Whether to save the results in matlab format in file "Fall.mat".
    :param float thresh_spks_perc: Threshold for denoising the deconvolved spike trains, in percentile. Any value in the ROIs-by-time-point deconvolved spike matrix that is below the matrix's xth percentile value is set to 0. Note that the same threshold is applied to all ROIs, and that this thresholding step does not binarize the deconvolved spike trains but simply filters out the low-amplitude spike events.
    :param float viz_vmin_perc: Minimum value for the colormap range, as percentile of the FOV fluorescence.
    :param float viz_vmax_perc: Maximum value for the colormap range, as percentile of the FOV fluorescence.
    :param str viz_cmap: Colormap for plotting the FOV.
    :param bool viz_show_grid: Whether or not to show the grid on FOVs.
    :param float viz_ticks_step: Step for the x- and y-ticks.
    :param int viz_n_samp_cells: Number of sample cells for the cell extraction preview.
    :param int viz_random_seed: Random seed for selecting sample cells.
    :param bool viz_show_all_footprints: Whether or not to show footprints of non-sample cells on the cell footprint FOV image. If False, only footprints of sample cells are displayed.
    """
    suite2p_output_conversion(
        fluo_file=fluo_file,
        neuropil_fluo_file=neuropil_fluo_file,
        spks_file=spks_file,
        stat_file=stat_file,
        ops_file=ops_file,
        iscell_file=iscell_file,
        save_npy=save_npy,
        save_isxd=save_isxd,
        save_NWB=save_NWB,
        save_mat=save_mat,
        thresh_spks_perc=thresh_spks_perc,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
        viz_n_samp_cells=viz_n_samp_cells,
        viz_random_seed=viz_random_seed,
        viz_show_all_footprints=viz_show_all_footprints,
    )

    try:
        logger.info("Registering output data")
        output_prefix = outputs.input_paths_to_output_prefix(
            fluo_file, neuropil_fluo_file, spks_file, stat_file, ops_file, iscell_file, max_name_len=100
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

