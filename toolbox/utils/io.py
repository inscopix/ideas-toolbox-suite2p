from datetime import datetime
from glob import glob
import isx
import logging
import numpy as np
import os
from tarfile import TarFile
from tifffile import tifffile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

logger = logging.getLogger()


def npy_to_isxd(npy_dir, output_dir, thresh_spks_perc, custom_fnames=None):
    """
    Convert native suite2p output into Inscopix isxd cellset and eventset.
    """
    # handle optional custom fnames
    if custom_fnames is not None:
        fname_list = custom_fnames
    else:
        fname_list = [
            f"{x}.npy" for x in ["F", "spks", "stat", "ops", "iscell"]
        ]

    # load native suite2p output files
    F = np.load(f"{npy_dir}/{fname_list[0]}", allow_pickle=True)
    spks = np.load(f"{npy_dir}/{fname_list[1]}", allow_pickle=True)
    stat = np.load(f"{npy_dir}/{fname_list[2]}", allow_pickle=True)
    ops = np.load(f"{npy_dir}/{fname_list[3]}", allow_pickle=True).item()
    iscell = np.load(f"{npy_dir}/{fname_list[4]}", allow_pickle=True)

    # if allow_overlap is False, then fluorescence traces of overlapping footprints are set to all zeros; the code below removes data from overlapping ROIs from preview figures
    if not ops["allow_overlap"]:
        idx_ok = np.where([len(np.unique(f)) > 1 for f in F])[0]
        F = F[idx_ok, :]
        spks = spks[idx_ok, :]
        stat = stat[idx_ok]
        iscell = iscell[idx_ok]

    # get cellset/eventset parameters
    num_cell, num_samples = F.shape
    num_pixels = ops["meanImg"].shape

    period_s = 1 / ops["fs"]
    num, den = period_s.as_integer_ratio()
    period = isx.Duration._from_num_den(num, den)
    if "start_time" in ops.keys() and isinstance(
        ops["start_time"], np.ndarray
    ):
        start = isx.Time._from_secs_since_epoch(
            isx.Duration.from_usecs(int(ops["start_time"].astype(int)))
        )
        timing = isx.Timing(
            num_samples=num_samples, period=period, start=start
        )
    else:
        timing = isx.Timing(num_samples=num_samples, period=period)
        logger.warning(
            "Date missing from the input data. Setting the start time of the output file to the Unix epoch (1970/01/01 00:00:00)."
        )

    spacing = isx.Spacing(num_pixels=num_pixels)

    names = [f"C{x:03}" for x in range(num_cell)]

    status_dict = {0: "rejected", 1: "accepted"}

    # write cellset file
    output_cs = f"{output_dir}/cellset_raw.isxd"
    cs_out = isx.CellSet.write(
        file_path=output_cs, timing=timing, spacing=spacing
    )
    for idx in range(num_cell):
        image = np.zeros(num_pixels, dtype="float32")
        ypix = stat[idx]["ypix"]
        xpix = stat[idx]["xpix"]
        image[ypix, xpix] = stat[idx]["lam"]
        image[image < 0] = 0
        image /= image.sum()

        trace = F[idx, :]

        name = names[idx]

        cs_out.set_cell_data(index=idx, image=image, trace=trace, name=name)

    for idx in range(num_cell):
        status = status_dict[iscell[idx, 0]]
        cs_out.set_cell_status(index=idx, status=status)

    # write eventset file
    cs = isx.CellSet.read(output_cs)
    offsets = np.array(
        [x.to_usecs() for x in cs.timing.get_offsets_since_start()], np.uint64
    )
    thresh_spks = np.percentile(spks, thresh_spks_perc)

    output_es = f"{output_dir}/eventset.isxd"
    es_out = isx.EventSet.write(
        file_path=output_es,
        timing=timing,
        cell_names=names,
    )
    for idx in range(num_cell):
        spike_train = spks[idx, :]
        idx_spk = spike_train > thresh_spks
        es_out.set_cell_data(
            index=idx,
            offsets=offsets[idx_spk],
            amplitudes=spike_train[idx_spk],
        )
    es_out.flush()


def extract_bruker2p_file(raw_movie_files, file_ext, data_dir):
    """
    Extract .zip or .tag.gz Bruker2P data and return its sampling rate and start time.
    """
    auth_ext_list = [".xml", ".env", ".ome", ".ome.tif"]
    for raw_movie_file in raw_movie_files:
        if file_ext == ".zip":
            with ZipFile(raw_movie_file, "r") as f:
                for member_info in f.infolist():
                    if member_info.is_dir():
                        continue
                    member_info.filename = os.path.basename(
                        member_info.filename
                    )
                    if any(
                        [
                            member_info.filename.endswith(ext)
                            for ext in auth_ext_list
                        ]
                    ):
                        f.extract(member_info, data_dir)
        elif file_ext == ".tar.gz":
            with TarFile.open(raw_movie_file, "r") as f:
                for member_info in f.getmembers():
                    if member_info.isdir():
                        continue
                    member_info.name = os.path.basename(member_info.name)
                    if any(
                        [
                            member_info.filename.endswith(ext)
                            for ext in auth_ext_list
                        ]
                    ):
                        f.extract(member_info, data_dir)

    # read XML to get version and fs
    xml_files = glob(data_dir + "*.xml")
    if len(xml_files) == 0:
        raise FileExistsError(
            f"No .xml file was found in {os.path.basename(raw_movie_files[0])}. Please make sure your Bruker2P data is complete."
        )
    elif len(xml_files) == 1:
        xml_file = xml_files[0]
    else:
        logger.info(f"Detected {len(xml_files)} .xml files: {xml_files}.")
        ometif_files = glob(data_dir + "*.ome.tif")
        idx_same_name = np.where(
            [
                any([x.split(".")[0] in y for y in ometif_files])
                for x in xml_files
            ]
        )[0]
        if len(idx_same_name) > 0:
            xml_file = xml_files[idx_same_name[0]]
        else:
            xml_file = xml_files[0]
        logger.info(f"Selected {xml_file}.")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bruker_version = root.attrib.get("version")
    logger.info(f"Processing Bruker 2p data v{bruker_version}...")
    fs = 1 / float(
        root.findall('.//PVStateValue/[@key="framePeriod"]')[0].attrib.get(
            "value"
        )
    )
    logger.info(f"Got sampling rate from xml: {fs} Hz")
    if "date" in root.attrib:
        date = root.attrib["date"]
        start_time = datetime.strptime(date, "%m/%d/%Y %I:%M:%S %p")
        # converting `start_time` to a NumPy array of dtype np.datetime64, otherwise it cannot be saved as a .mat file
        start_time = np.array(start_time, dtype=np.datetime64)
        logger.info(f"Got date of recording from xml: {date}")
    else:
        start_time = None
        logger.warning(
            "Date missing from the input data. Will set start time to the Unix epoch (1970/01/01 00:00:00)."
        )

    return fs, start_time


def save_local_corr_img(ops, output_dir):
    """
    Save the local correlation image as a standalone .tif file, e.g., for further use as template image in Multi-Session Registration.
    """
    img = np.zeros((ops["Ly"], ops["Lx"]), dtype=np.float32)
    img[
        ops["yrange"][0] : ops["yrange"][1],
        ops["xrange"][0] : ops["xrange"][1],
    ] = ops["Vcorr"]
    output_path = f"{output_dir}/local_corr_img.tif"
    tifffile.imwrite(output_path, img)
    logger.info("Saved the local correlation image!")

    return output_path
