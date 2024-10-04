import isx
import numpy as np


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

    # get cellset/eventset parameters
    num_cell, num_samples = F.shape
    num_pixels = ops["meanImg"].shape

    period = isx.Duration.from_usecs(1e6 / ops["fs"])
    timing = isx.Timing(num_samples=num_samples, period=period)

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
