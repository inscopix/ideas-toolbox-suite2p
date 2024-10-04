import os
from pynwb import NWBHDF5IO
from dandi.validate import validate
from dandi.validate_types import Severity
import logging

logger = logging.getLogger()


def is_dandi_compliant(input_nwb_filename):
    """Determine if a given NWB file is DANDI compliant.

    Returns True if compliant, False if non-compliant, and None when unable to determine DANDI compliance.

    :param input_nwb_filename: path to the nwb file
    """
    dandi_compliant = True
    try:
        for v in validate(input_nwb_filename):
            if v.severity == Severity.ERROR:
                if v.message != "Path is not inside a Dandiset":
                    logger.warning(
                        f"File '{os.path.basename(input_nwb_filename)}' is not DANDI compliant. {v.message}"
                    )
                    dandi_compliant = False
    except Exception:
        logger.warning(
            "Could not determine if the nwb file is DANDI compliant"
        )
        dandi_compliant = None

    # validate external file references
    supported_video_file_extensions = [
        "mp4",
        "avi",
        "wmv",
        "mov",
        "flv",
        "mkv",
    ]
    with NWBHDF5IO(input_nwb_filename, mode="r") as io:
        nwb = io.read()
        if "ImageSeries" in nwb.acquisition:
            external_files = (
                nwb.acquisition["ImageSeries"].external_file[:].tolist()
            )
            for f in external_files:
                ext = f.lower().split(".")[-1]
                if ext not in supported_video_file_extensions:
                    logger.warn(
                        f"File '{os.path.basename(input_nwb_filename)}' is not DANDI compliant."
                        f" External file reference '{os.path.basename(f)}' is not supported."
                        f" The video file extensions supported by DANDI are"
                        f" {', '.join(supported_video_file_extensions)}."
                    )
                    dandi_compliant = False

    return dandi_compliant


def construct_nwb_file_metadata(nwb_filename):
    """Construct a dictionary of metadata for a given NWB file.

    :param nwb_filename: path to the nwb file
    """
    with NWBHDF5IO(nwb_filename, "r") as io:
        nwb_file = io.read()
        nwb_session_start_time = nwb_file.session_start_time
    metadata = {
        "dataset": {
            "session_start_time": nwb_session_start_time.strftime(
                "%m/%d/%Y %H:%M:%S.%f"
            ),
            "dandi_compliant": is_dandi_compliant(nwb_filename),
        }
    }
    return metadata
