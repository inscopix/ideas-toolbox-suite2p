from glob import glob
import isx
import logging
from natsort import natsorted
import numpy as np
import os
import shutil
from suite2p import (
    default_ops,
    io,
    registration,
    detection,
    classification,
    extraction,
)
from tarfile import TarFile
import time
from toolbox.utils import io as tlbxio
from toolbox.utils import metadata, preview, utilities
import xml.etree.ElementTree as ET
from zipfile import ZipFile


logger = logging.getLogger()


def suite2p_binary_conversion(
    raw_movie_files,
    nplanes=1,
    nchannels=1,
    functional_chan=1,
    fs=None,
    bruker_bidirectional=False,
):
    """
    Tool to convert raw 2P input movie(s) into a suite2p binary file. This constitutes the first step of the suite2p end-to-end pipeline.
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
    file_ext = "." + ".".join(
        os.path.basename(raw_movie_files[0]).split(".")[1:]
    )
    if file_ext == ".isxd":
        ops["input_format"] = "isxd"
        fs_auto = (
            1e6 / isx.Movie.read(raw_movie_files[0]).timing.period.to_usecs()
        )
    elif file_ext in [".zip", ".tar.gz"]:
        logger.info(
            f"Bruker Ultima 2P {file_ext} movie detected: setting `ops['bruker']` to `True`."
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
        ops["input_format"] = "bruker"
        tif_list = glob(f"{data_dir}/*tif")
        channel_list = [
            int(os.path.basename(x).split("_")[2][-1]) for x in tif_list
        ]
        ops["functional_chan"] = channel_list[0]

        xml_file = glob(data_dir + "*.xml")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bruker_version = root.attrib.get("version")
        fs_auto = 1 / float(
            root.findall('.//PVStateValue/[@key="framePeriod"]')[0].attrib.get(
                "value"
            )
        )
    elif file_ext in [".tif", ".tiff"]:
        ops["input_format"] = "tif"
        fs_auto = ops["fs"]
    else:
        raise ValueError(
            f"File format {file_ext} not recognized as either Inscopix .isxd, Bruker Ultima 2P .zip or .tar.gz, or standard .tif/.tiff stack."
        )

    # curate parameters
    if fs is not None:
        ops["fs"] = fs
    else:
        ops["fs"] = fs_auto
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
    print(
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
    shutil.move(
        ops0["ops_path"], f"{ideas_output_dir}/ops_binary_conversion.npy"
    )
    if os.path.exists(f"{ideas_output_dir}/suite2p/"):
        shutil.rmtree(f"{ideas_output_dir}/suite2p/")
    print("ALL DONE!")


def suite2p_registration(
    raw_binary_file,
    ops_file,
    frames_include=-1,
    align_by_chan=1,
    nimg_init=300,
    batch_size=500,
    maxregshift=0.1,
    smooth_sigma=1.15,
    smooth_sigma_time=0,
    two_step_registration=False,
    subpixel=10,
    th_badframes=1.0,
    norm_frames=True,
    force_refImg=False,
    pad_fft=False,
    one_p_reg=False,
    spatial_hp_reg=42,
    pre_smooth=0,
    spatial_taper=40,
    nonrigid=True,
    block_size=[128, 128],
    snr_thresh=1.2,
    maxregshiftNR=5,
    do_bidiphase=False,
    bidiphase=0,
    bidi_corrected=False,
    viz_vmin_perc=0,
    viz_vmax_perc=99,
    viz_cmap="plasma",
    viz_show_grid=True,
    viz_ticks_step=128,
    viz_display_rate=10,
):
    """
    Tool to run suite2p registration on a raw suite2p binary movie. This constitutes the second step of the suite2p end-to-end pipeline.
    """
    # define registered binary movie name
    ideas_output_dir = os.getcwd()
    reg_binary_path = f"{ideas_output_dir}/data.bin"
    ops_path = f"{ideas_output_dir}/ops_registration.npy"

    # load input parameter file
    ops = np.load(ops_file[0], allow_pickle=True).item()
    ops = utilities.set_hardcoded_parameters(ops)
    ops["raw_file"] = raw_binary_file[0]
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
    f_raw = io.BinaryFile(Ly=Ly, Lx=Lx, filename=ops["raw_file"])
    f_reg = io.BinaryFile(
        Ly=Ly, Lx=Lx, filename=ops["reg_file"], n_frames=f_raw.shape[0]
    )  # Set registered binary file to have same n_frames

    # [start of suite2p code]
    # REGISTRATION
    t11 = time.time()
    plane_times = {}
    print("----------- REGISTRATION")
    refImg = (
        ops["refImg"]
        if "refImg" in ops and ops.get("force_refImg", False)
        else None
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

    ops = registration.save_registration_outputs_to_ops(
        registration_outputs, ops
    )
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
    print("----------- Total %0.2f sec" % plane_times["registration"])
    n_frames, Ly, Lx = f_reg.shape

    if ops["two_step_registration"] and ops["keep_movie_raw"]:
        print("----------- REGISTRATION STEP 2")
        print("(making mean image (excluding bad frames)")
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
        print(
            "----------- Total %0.2f sec"
            % plane_times["two_step_registration"]
        )

    # compute metrics for registration
    if ops.get("do_regmetrics", True) and n_frames >= 1500:
        t0 = time.time()
        # n frames to pick from full movie
        nsamp = min(
            2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames
        )
        inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
        mov = f_reg[inds]
        mov = mov[
            :,
            ops["yrange"][0] : ops["yrange"][-1],
            ops["xrange"][0] : ops["xrange"][-1],
        ]
        ops = registration.get_pc_metrics(mov, ops)
        plane_times["registration_metrics"] = time.time() - t0
        print(
            "Registration metrics, %0.2f sec."
            % plane_times["registration_metrics"]
        )
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

    print("ALL DONE!")


def suite2p_roi_detection(
    reg_binary_file,
    ops_file,
    classifier_path=None,
    tau=1.0,
    sparse_mode=True,
    spatial_scale=0,
    connected=True,
    threshold_scaling=1,
    spatial_hp_detect=25,
    max_overlap=0.75,
    high_pass=100,
    smooth_masks=True,
    max_iterations=20,
    nbinned=5000,
    denoise=False,
    anatomical_only=0,
    diameter=0,
    cellprob_threshold=0.0,
    flow_threshold=1.5,
    spatial_hp_cp=0,
    pretrained_model="cyto",
    preclassify=0.0,
    chan2_thres=0.65,
    viz_vmin_perc=0,
    viz_vmax_perc=99,
    viz_cmap="plasma",
    viz_show_grid=True,
    viz_ticks_step=128,
):
    """
    Tool to run suite2p ROI detection on a registered suite2p binary movie. This constitutes the third step of the suite2p end-to-end pipeline.
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
        print(f"NOTE: applying classifier {str(ops_classfile)}")
        classfile = ops_classfile
    elif ops["use_builtin_classifier"] or not user_classfile.is_file():
        print(f"NOTE: Applying builtin classifier at {str(builtin_classfile)}")
        classfile = builtin_classfile
    else:
        print(f"NOTE: applying default {str(user_classfile)}")
        classfile = user_classfile

    # CELL DETECTION
    t11 = time.time()
    plane_times = {}
    print("----------- ROI DETECTION")
    ops, stat = detection.detection_wrapper(
        f_reg, ops=ops, classfile=classfile
    )
    plane_times["detection"] = time.time() - t11
    print("----------- Total %0.2f sec." % plane_times["detection"])
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

    print("ALL DONE!")


def suite2p_roi_extraction(
    reg_binary_file,
    stat_file,
    ops_file,
    neuropil_extract=True,
    allow_overlap=False,
    min_neuropil_pixels=350,
    inner_neuropil_radius=2,
    lam_percentile=50,
    viz_show_grid=True,
    viz_ticks_step=128,
    viz_n_samp_cells=20,
    viz_random_seed=0,
    viz_show_all_footprints=True,
):
    """
    Tool to run suite2p ROI extraction on a registered suite2p binary movie, using the previously detected ROIs. This constitutes the fourth step of the suite2p end-to-end pipeline.
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
    print("----------- EXTRACTION")
    f_reg_chan2 = None
    stat, F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(
        stat, f_reg, f_reg_chan2=f_reg_chan2, ops=ops
    )

    plane_times["extraction"] = time.time() - t11
    print("----------- Total %0.2f sec." % plane_times["extraction"])
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

    print("ALL DONE!")


def suite2p_roi_classification(
    stat_file,
    ops_file,
    classifier_path=None,
    soma_crop=60,
    viz_vmin_perc=0,
    viz_vmax_perc=99,
    viz_cmap="plasma",
    viz_show_grid=True,
    viz_ticks_step=128,
):
    """
    Tool to run suite2p ROI classification on the extracted ROIs. This constitutes the fifth step of the suite2p end-to-end pipeline.
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
        print(f"NOTE: applying classifier {str(ops_classfile)}")
        classfile = ops_classfile
    elif ops["use_builtin_classifier"] or not user_classfile.is_file():
        print(f"NOTE: Applying builtin classifier at {str(builtin_classfile)}")
        classfile = builtin_classfile
    else:
        print(f"NOTE: applying default {str(user_classfile)}")
        classfile = user_classfile

    # ROI CLASSIFICATION
    t11 = time.time()
    plane_times = {}
    print("----------- CLASSIFICATION")
    if len(stat):
        iscell = classification.classify(stat=stat, classfile=classfile)
    else:
        iscell = np.zeros((0, 2))
    plane_times["classification"] = time.time() - t11
    print("----------- Total %0.2f sec." % plane_times["classification"])
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
    preview.create_output_previews(
        ops=ops,
        steps="roi_classification",
        vmin_perc=viz_vmin_perc,
        vmax_perc=viz_vmax_perc,
        cmap=viz_cmap,
        show_grid=viz_show_grid,
        ticks_step=int(viz_ticks_step),
    )

    print("ALL DONE!")


def suite2p_spike_deconvolution(
    fluo_file,
    neuropil_fluo_file,
    ops_file,
    tau=1.0,
    neucoeff=0.7,
    baseline="maximin",
    win_baseline=60.0,
    sig_baseline=10.0,
    prctile_baseline=8.0,
    viz_n_samp_cells=20,
    viz_random_seed=0,
    viz_show_all_footprints=True,
):
    """
    Tool to run suite2p spike deconvolution on the extracted fluorescence traces. This constitutes the sixth step of the suite2p end-to-end pipeline.
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
    print("----------- SPIKE DECONVOLUTION")
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
    print("----------- Total %0.2f sec." % plane_times["deconvolution"])
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
    preview.create_output_previews(
        ops=ops,
        steps="spike_deconvolution",
        n_samp_cells=int(viz_n_samp_cells),
        random_seed=int(viz_random_seed),
        show_all_footprints=viz_show_all_footprints,
    )

    print("ALL DONE!")


def suite2p_output_conversion(
    fluo_file,
    neuropil_fluo_file,
    spks_file,
    stat_file,
    ops_file,
    iscell_file,
    save_npy=True,
    save_isxd=True,
    save_NWB=False,
    save_mat=False,
    thresh_spks_perc=99.7,
    viz_vmin_perc=0,
    viz_vmax_perc=99,
    viz_cmap="plasma",
    viz_show_grid=True,
    viz_ticks_step=128,
    viz_n_samp_cells=20,
    viz_random_seed=0,
    viz_show_all_footprints=True,
):
    """
    Tool to output suite2p results in specific formats. This constitutes the seventh and last step of the suite2p end-to-end pipeline.
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
        print()
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
            shutil.copyfile(
                file[0], os.path.join(plane_output_dir, fixed_fname)
            )
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
