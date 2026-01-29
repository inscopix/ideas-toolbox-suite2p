import os
import numpy as np

# from zipfile import ZipFile
import pytest
import shutil
from toolbox.tools import suite2p_individual_steps as s2pis
from suite2p import io

data_dir = "/ideas/data"


@pytest.mark.parametrize(
    "raw_movie_files,nplanes,nchannels,functional_chan,bruker_bidirectional,expected_files,expected_shape,expected_range,expected_mean_first_last,expected_fs,expected_mean_img,",
    [
        [
            ["sample_300x512x512_movie.isxd"],
            1,
            1,
            1,
            False,
            ["data_raw.bin", "ops_binary_conversion.npy"],
            (300, 512, 512),
            (0, 8191),
            325.29364585876465,
            29.87393200693075,
            34463780.0,
        ]
    ],
)
def test_suite2p_binary_conversion(
    raw_movie_files,
    nplanes,
    nchannels,
    functional_chan,
    bruker_bidirectional,
    expected_files,
    expected_shape,
    expected_range,
    expected_mean_first_last,
    expected_fs,
    expected_mean_img,
):
    """
    Test that suite2p_binary_conversion() runs properly and outputs the expected file.
    """
    raw_movie_files = [f"{data_dir}/{x}" for x in raw_movie_files]

    for idx, f in enumerate(raw_movie_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        raw_movie_files[idx] = dest

    s2pis.suite2p_binary_conversion(
        raw_movie_files=raw_movie_files,
        nplanes=nplanes,
        nchannels=nchannels,
        functional_chan=functional_chan,
        bruker_bidirectional=bruker_bidirectional,
    )

    # ensure data_raw.bin and ops_binary_conversion.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check data_raw.bin
    mov = io.BinaryFile(
        Ly=expected_shape[2],
        Lx=expected_shape[1],
        filename=expected_files[0],
    )

    assert mov.shape == expected_shape, "Unexpected movie shape!"

    mov_range = (np.min(mov.data), np.max(mov.data))
    assert np.allclose(mov_range, expected_range), "Unexpected movie range!"

    mov_mean_first_last = np.mean([mov.data[0, :, :], mov.data[-1, :, :]])
    assert np.allclose(
        mov_mean_first_last, expected_mean_first_last
    ), "Unexpected movie frames!"

    # check ops_binary_conversion.npy
    ops = np.load(expected_files[1], allow_pickle=True).item()
    assert np.allclose(ops["fs"], expected_fs), "Unexpected sampling rate!"

    assert (
        np.sum(ops["meanImg"]) == expected_mean_img
    ), "Unexpected mean image!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "raw_binary_file,ops_file,frames_include,align_by_chan,nimg_init,batch_size,maxregshift,smooth_sigma,smooth_sigma_time,two_step_registration,subpixel,th_badframes,norm_frames,force_refImg,pad_fft,one_p_reg,spatial_hp_reg,pre_smooth,spatial_taper,nonrigid,block_size,snr_thresh,maxregshiftNR,do_bidiphase,bidiphase,bidi_corrected,viz_vmin_perc,viz_vmax_perc,viz_cmap,viz_show_grid,viz_ticks_step,viz_display_rate,expected_files,expected_shape,expected_range,expected_mean_first_last,expected_fs,expected_sum_xyoff,",
    [
        [
            "data_raw.bin",
            "ops_binary_conversion.npy",
            -1,
            1,
            300,
            500,
            0.1,
            1.15,
            0,
            False,
            10,
            1.0,
            True,
            False,
            False,
            False,
            42,
            0,
            40,
            True,
            [128, 128],
            1.2,
            5,
            False,
            0,
            False,
            0,
            99,
            "plasma",
            True,
            128,
            10,
            ["data.bin", "ops_registration.npy"],
            (300, 512, 512),
            (-829, 8186),
            324.7294921875,
            29.87393200693075,
            9,
        ]
    ],
)
def test_suite2p_registration(
    raw_binary_file,
    ops_file,
    frames_include,
    align_by_chan,
    nimg_init,
    batch_size,
    maxregshift,
    smooth_sigma,
    smooth_sigma_time,
    two_step_registration,
    subpixel,
    th_badframes,
    norm_frames,
    force_refImg,
    pad_fft,
    one_p_reg,
    spatial_hp_reg,
    pre_smooth,
    spatial_taper,
    nonrigid,
    block_size,
    snr_thresh,
    maxregshiftNR,
    do_bidiphase,
    bidiphase,
    bidi_corrected,
    viz_vmin_perc,
    viz_vmax_perc,
    viz_cmap,
    viz_show_grid,
    viz_ticks_step,
    viz_display_rate,
    expected_files,
    expected_shape,
    expected_range,
    expected_mean_first_last,
    expected_fs,
    expected_sum_xyoff,
):
    """
    Test that suite2p_registration() runs properly, outputs the expected file and performs motion correction as expected.
    """
    input_files = [f"{data_dir}/{x}" for x in [raw_binary_file, ops_file]]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_registration(
        raw_binary_file=[input_files[0]],
        ops_file=[input_files[1]],
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

    # ensure data.bin and ops_registration.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check data.bin
    mov = io.BinaryFile(
        Ly=expected_shape[2],
        Lx=expected_shape[1],
        filename=expected_files[0],
    )

    assert mov.shape == expected_shape, "Unexpected movie shape!"

    mov_range = (np.min(mov.data), np.max(mov.data))
    assert np.allclose(mov_range, expected_range), "Unexpected movie range!"

    mov_mean_first_last = np.mean([mov.data[0, :, :], mov.data[-1, :, :]])
    assert np.allclose(
        mov_mean_first_last, expected_mean_first_last
    ), "Unexpected movie frames!"

    # check ops_registration.npy
    ops = np.load(expected_files[1], allow_pickle=True).item()
    assert np.allclose(ops["fs"], expected_fs), "Unexpected sampling rate!"

    assert (
        np.sum(ops["xoff"]) + np.sum(ops["yoff"]) == expected_sum_xyoff
    ), "Unexpected sum of xoff and yoff!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "reg_binary_file,ops_file,classifier_path,tau,sparse_mode,spatial_scale,connected,threshold_scaling,spatial_hp_detect,max_overlap,high_pass,smooth_masks,max_iterations,nbinned,denoise,anatomical_only,diameter,cellprob_threshold,flow_threshold,spatial_hp_cp,pretrained_model,preclassify,chan2_thres,viz_vmin_perc,viz_vmax_perc,viz_cmap,viz_show_grid,viz_ticks_step,expected_files,expected_n_rois,expected_sum_npix_soma,",
    [
        [
            "data.bin",
            "ops_registration.npy",
            None,
            1.0,
            True,
            0,
            True,
            1,
            25,
            0.75,
            100,
            True,
            20,
            5000,
            False,
            0,
            0,
            0.0,
            1.5,
            0,
            "cyto",
            0.0,
            0.65,
            0,
            99,
            "plasma",
            True,
            128,
            ["stat_ROI_detection.npy", "ops_ROI_detection.npy"],
            54,
            3251,
        ]
    ],
)
def test_suite2p_roi_detection(
    reg_binary_file,
    ops_file,
    classifier_path,
    tau,
    sparse_mode,
    spatial_scale,
    connected,
    threshold_scaling,
    spatial_hp_detect,
    max_overlap,
    high_pass,
    smooth_masks,
    max_iterations,
    nbinned,
    denoise,
    anatomical_only,
    diameter,
    cellprob_threshold,
    flow_threshold,
    spatial_hp_cp,
    pretrained_model,
    preclassify,
    chan2_thres,
    viz_vmin_perc,
    viz_vmax_perc,
    viz_cmap,
    viz_show_grid,
    viz_ticks_step,
    expected_files,
    expected_n_rois,
    expected_sum_npix_soma,
):
    """
    Test that suite2p_roi_detection() runs properly, outputs the expected file and performs ROI detection as expected.
    """
    input_files = [f"{data_dir}/{x}" for x in [reg_binary_file, ops_file]]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_roi_detection(
        reg_binary_file=[input_files[0]],
        ops_file=[input_files[1]],
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

    # ensure stat_ROI_detection.npy and ops_ROI_detection.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check stat_ROI_detection.npy
    stat = np.load(expected_files[0], allow_pickle=True)

    assert len(stat) == expected_n_rois, "Unexpected number of detected ROIs!"

    assert (
        sum([x["npix_soma"] for x in stat]) == expected_sum_npix_soma
    ), "Unexpected sum of somatic areas across all ROIs!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "reg_binary_file,stat_file,ops_file,neuropil_extract,allow_overlap,min_neuropil_pixels,inner_neuropil_radius,lam_percentile,viz_show_grid,viz_ticks_step,viz_n_samp_cells,viz_random_seed,viz_show_all_footprints,expected_files,expected_n_rois,expected_sum_npix_soma,expected_mat_shape,expected_mean_F,expected_mean_Fneu,",
    [
        [
            "data.bin",
            "stat_ROI_detection.npy",
            "ops_ROI_detection.npy",
            True,
            False,
            350,
            2,
            50,
            True,
            128,
            20,
            0,
            True,
            ["stat.npy", "F.npy", "Fneu.npy", "ops_ROI_extraction.npy"],
            54,
            3251,
            (54, 300),
            523.0839,
            391.02588,
        ]
    ],
)
def test_suite2p_roi_extraction(
    reg_binary_file,
    stat_file,
    ops_file,
    neuropil_extract,
    allow_overlap,
    min_neuropil_pixels,
    inner_neuropil_radius,
    lam_percentile,
    viz_show_grid,
    viz_ticks_step,
    viz_n_samp_cells,
    viz_random_seed,
    viz_show_all_footprints,
    expected_files,
    expected_n_rois,
    expected_sum_npix_soma,
    expected_mat_shape,
    expected_mean_F,
    expected_mean_Fneu,
):
    """
    Test that suite2p_roi_extraction() runs properly, outputs the expected file and performs ROI extraction as expected.
    """
    input_files = [
        f"{data_dir}/{x}" for x in [reg_binary_file, stat_file, ops_file]
    ]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_roi_extraction(
        reg_binary_file=[input_files[0]],
        stat_file=[input_files[1]],
        ops_file=[input_files[2]],
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

    # ensure stat.npy, F.npy, Fneu.npy and ops_ROI_extraction.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check stat.npy
    stat = np.load(expected_files[0], allow_pickle=True)
    assert len(stat) == expected_n_rois, "Unexpected number of extracted ROIs!"

    assert (
        sum([x["npix_soma"] for x in stat]) == expected_sum_npix_soma
    ), "Unexpected sum of somatic areas across all ROIs!"

    # check F.npy
    F = np.load(expected_files[1], allow_pickle=True)
    assert (
        F.shape == expected_mat_shape
    ), "Unexpected shape of extracted traces matrix!"

    assert np.allclose(
        np.nanmean(F), expected_mean_F
    ), "Unexpected mean value of fluorescence traces!"

    # check Fneu.npy
    Fneu = np.load(expected_files[2], allow_pickle=True)
    assert (
        Fneu.shape == expected_mat_shape
    ), "Unexpected shape of extracted traces matrix!"

    assert np.allclose(
        np.nanmean(Fneu), expected_mean_Fneu
    ), "Unexpected mean value of neuropil fluorescence traces!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "stat_file,ops_file,classifier_path,soma_crop,viz_vmin_perc,viz_vmax_perc,viz_cmap,viz_show_grid,viz_ticks_step,expected_files,expected_n_rois,expected_n_accepted,expected_mean_prob,",
    [
        [
            "stat.npy",
            "ops_ROI_extraction.npy",
            None,
            60,
            0,
            99,
            "plasma",
            True,
            128,
            ["iscell.npy", "ops_ROI_classification.npy"],
            54,
            19,
            0.3588041016285435,
        ]
    ],
)
def test_suite2p_roi_classification(
    stat_file,
    ops_file,
    classifier_path,
    soma_crop,
    viz_vmin_perc,
    viz_vmax_perc,
    viz_cmap,
    viz_show_grid,
    viz_ticks_step,
    expected_files,
    expected_n_rois,
    expected_n_accepted,
    expected_mean_prob,
):
    """
    Test that suite2p_roi_classification() runs properly, outputs the expected file and performs ROI classification as expected.
    """
    input_files = [f"{data_dir}/{x}" for x in [stat_file, ops_file]]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_roi_classification(
        stat_file=[input_files[0]],
        ops_file=[input_files[1]],
        classifier_path=classifier_path,
        soma_crop=soma_crop,
        viz_vmin_perc=viz_vmin_perc,
        viz_vmax_perc=viz_vmax_perc,
        viz_cmap=viz_cmap,
        viz_show_grid=viz_show_grid,
        viz_ticks_step=viz_ticks_step,
    )

    # ensure iscell.npy and ops_ROI_classification.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check iscell.npy
    iscell = np.load(expected_files[0], allow_pickle=True)
    assert (
        len(iscell) == expected_n_rois
    ), "Unexpected number of extracted ROIs!"

    assert (
        np.sum(iscell[:, 0]).astype(int) == expected_n_accepted
    ), "Unexpected number of accepted cells!"

    assert np.allclose(
        np.mean(iscell[:, 1]), expected_mean_prob
    ), "Unexpected mean probability across ROIs!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "fluo_file,neuropil_fluo_file,ops_file,tau,neucoeff,baseline,win_baseline,sig_baseline,prctile_baseline,viz_n_samp_cells,viz_random_seed,viz_show_all_footprints,expected_files,expected_mat_shape,expected_spks_sum,",
    [
        [
            "F.npy",
            "Fneu.npy",
            "ops_ROI_extraction.npy",
            1.0,
            0.7,
            "maximin",
            60.0,
            10.0,
            8.0,
            20,
            0,
            True,
            ["spks.npy", "ops_spike_deconvolution.npy"],
            (54, 300),
            45205.14,
        ]
    ],
)
def test_suite2p_spike_deconvolution(
    fluo_file,
    neuropil_fluo_file,
    ops_file,
    tau,
    neucoeff,
    baseline,
    win_baseline,
    sig_baseline,
    prctile_baseline,
    viz_n_samp_cells,
    viz_random_seed,
    viz_show_all_footprints,
    expected_files,
    expected_mat_shape,
    expected_spks_sum,
):
    """
    Test that suite2p_spike_deconvolution() runs properly, outputs the expected file and performs spike deconvolution as expected.
    """
    input_files = [
        f"{data_dir}/{x}" for x in [fluo_file, neuropil_fluo_file, ops_file]
    ]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_spike_deconvolution(
        fluo_file=[input_files[0]],
        neuropil_fluo_file=[input_files[1]],
        ops_file=[input_files[2]],
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

    # ensure spks.npy and ops_spike_deconvolution.npy files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # check spks.npy
    spks = np.load(expected_files[0], allow_pickle=True)
    assert (
        spks.shape == expected_mat_shape
    ), "Unexpected shape of deconvolved spikes matrix!"

    assert np.allclose(
        np.sum(spks), expected_spks_sum
    ), "Unexpected deconvolved spikes!"

    # clean up
    for f in expected_files:
        os.remove(f)


@pytest.mark.parametrize(
    "fluo_file,neuropil_fluo_file,spks_file,stat_file,ops_file,iscell_file,save_npy,save_isxd,save_NWB,save_mat,thresh_spks_perc,viz_vmin_perc,viz_vmax_perc,viz_cmap,viz_show_grid,viz_ticks_step,viz_n_samp_cells,viz_random_seed,viz_show_all_footprints,expected_files,",
    [
        [
            "F.npy",
            "Fneu.npy",
            "spks.npy",
            "stat.npy",
            "ops_spike_deconvolution.npy",
            "iscell.npy",
            True,
            True,
            True,
            True,
            99.7,
            0,
            99,
            "plasma",
            True,
            128,
            20,
            0,
            True,
            [
                "suite2p_output.zip",
                "cellset_raw.isxd",
                "eventset.isxd",
                "ophys.nwb",
                "Fall.mat",
            ],
        ]
    ],
)
def test_suite2p_output_conversion(
    fluo_file,
    neuropil_fluo_file,
    spks_file,
    stat_file,
    ops_file,
    iscell_file,
    save_npy,
    save_isxd,
    save_NWB,
    save_mat,
    thresh_spks_perc,
    viz_vmin_perc,
    viz_vmax_perc,
    viz_cmap,
    viz_show_grid,
    viz_ticks_step,
    viz_n_samp_cells,
    viz_random_seed,
    viz_show_all_footprints,
    expected_files,
):
    """
    Test that suite2p_spike_deconvolution() runs properly, outputs the expected file and performs spike deconvolution as expected.
    """
    input_files = [
        f"{data_dir}/{x}"
        for x in [
            fluo_file,
            neuropil_fluo_file,
            spks_file,
            stat_file,
            ops_file,
            iscell_file,
        ]
    ]

    for idx, f in enumerate(input_files):
        dest = os.path.join(os.getcwd(), os.path.basename(f))
        shutil.copy(f, dest)
        input_files[idx] = dest

    s2pis.suite2p_output_conversion(
        fluo_file=[input_files[0]],
        neuropil_fluo_file=[input_files[1]],
        spks_file=[input_files[2]],
        stat_file=[input_files[3]],
        ops_file=[input_files[4]],
        iscell_file=[input_files[5]],
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

    # ensure suite2p_output.zip, cellset_raw.isxd, eventset.isxd, ophys.nwb and Fall.mat files are present
    dir_list = os.listdir(os.getcwd())
    for f in expected_files:
        assert f in dir_list, f"{f} not in current working directory!"

    # clean up
    for f in expected_files:
        os.remove(f)
