import os
from toolbox.utils.nwb_utils import (
    is_dandi_compliant,
    construct_nwb_file_metadata,
)


INPUT_DIR = "/ideas/data/"


def test_dandi_compliant_data():
    """Validate DANDI-compliant NWB data and ensure it is recognized as such"""
    input_file = os.path.join(INPUT_DIR, "dandi_compliant_movie.nwb")
    assert is_dandi_compliant(input_file)


def test_non_dandi_compliant_data():
    """Validate NWB data that is not DANDI compliant and ensure it is recognized as such"""
    input_file = os.path.join(INPUT_DIR, "non_dandi_compliant_movie.nwb")
    assert not is_dandi_compliant(input_file)


def test_construct_nwb_file_metadata_for_dandi_compliant_data():
    """Ensure metadata is extracted properly for DANDI-compliant NWB data"""
    input_file = os.path.join(INPUT_DIR, "dandi_compliant_movie.nwb")
    act_metadata = construct_nwb_file_metadata(input_file)
    exp_metadata = {
        "dataset": {
            "session_start_time": "10/07/2019 16:13:46.418463",
            "dandi_compliant": True,
        }
    }
    assert act_metadata == exp_metadata


def test_construct_nwb_file_metadata_for_non_dandi_compliant_data():
    """Ensure metadata is extracted properly for NWB data that is not DANDI compliant"""
    input_file = os.path.join(INPUT_DIR, "non_dandi_compliant_movie.nwb")
    act_metadata = construct_nwb_file_metadata(input_file)
    exp_metadata = {
        "dataset": {
            "session_start_time": "10/07/2019 16:13:46.418463",
            "dandi_compliant": False,
        }
    }
    assert act_metadata == exp_metadata
