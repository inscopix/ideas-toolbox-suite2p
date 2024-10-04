import os
import numpy as np
from zipfile import ZipFile
import pytest
import shutil
from toolbox.tools.suite2p_wrapper import run_suite2p_end_to_end

data_dir = "/ideas/data"


@pytest.mark.parametrize(
    "raw_movie_files,ops_file,classifier_path,params_from,tau,frames_include,save_npy,save_isxd,save_NWB,save_mat,maxregshift,th_badframes,nonrigid,threshold_scaling,neucoeff,thresh_spks_perc,expected_ref_image,expected_F_shape,expected_F_mean,expected_spks_sum,expected_accepted_cells,",
    [
        [
            ["sample_300x512x512_movie.isxd"],
            None,
            None,
            "table",
            1.0,
            -1,
            True,
            True,
            False,
            False,
            0.1,
            1.0,
            True,
            1.0,
            0.7,
            99.7,
            68771274,
            (54, 300),
            523.0839,
            45205.14,
            19,
        ]
    ],
)
def test_run_suite2p_end_to_end(
    raw_movie_files,
    ops_file,
    classifier_path,
    params_from,
    tau,
    frames_include,
    save_npy,
    save_isxd,
    save_NWB,
    save_mat,
    maxregshift,
    th_badframes,
    nonrigid,
    threshold_scaling,
    neucoeff,
    thresh_spks_perc,
    expected_ref_image,
    expected_F_shape,
    expected_F_mean,
    expected_spks_sum,
    expected_accepted_cells,
):
    """
    Test that run_suite2p_end_to_end() runs properly and outputs the expected files.
    """
    raw_movie_files = [f"{data_dir}/{x}" for x in raw_movie_files]

    for idx, f in enumerate(raw_movie_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        raw_movie_files[idx] = dest

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
        maxregshift=maxregshift,
        th_badframes=th_badframes,
        nonrigid=nonrigid,
        threshold_scaling=threshold_scaling,
        neucoeff=neucoeff,
        thresh_spks_perc=thresh_spks_perc,
    )

    # ensure suite2p_output.zip file is present and contains all expected files
    dir_list = os.listdir(os.getcwd())
    assert (
        "suite2p_output.zip" in dir_list
    ), "suite2p_output.zip not in current working directory!"
    with ZipFile("suite2p_output.zip", "r") as f:
        for member_info in f.infolist():
            if member_info.is_dir():
                continue
            member_info.filename = os.path.basename(member_info.filename)
            f.extract(member_info)
    dir_list = os.listdir(os.getcwd())
    output_file_list = ["F", "Fneu", "spks", "stat", "ops", "iscell"]
    assert all(
        [f"{x}.npy" in dir_list for x in output_file_list]
    ), "Not all files are present!"

    # check registration
    ops = np.load("ops.npy", allow_pickle=True).item()
    assert (
        np.sum(ops["refImg"]) == expected_ref_image
    ), "Unexpected reference image!"

    # check neuropil-corrected fluorescence traces
    F = np.load("F.npy", allow_pickle=True)
    assert F.shape == expected_F_shape, "Unexpected F shape!"
    assert np.allclose(
        np.mean(F), expected_F_mean
    ), "Unexpected fluorescence traces!"

    # check spike sum
    spks = np.load("spks.npy", allow_pickle=True)
    assert np.allclose(
        np.sum(spks), expected_spks_sum
    ), "Unexpected deconvolved spikes!"

    # check classification labels
    iscell = np.load("iscell.npy", allow_pickle=True)
    assert (
        int(iscell[:, 0].sum()) == expected_accepted_cells
    ), "Unexpected cell labels!"
