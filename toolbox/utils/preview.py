from colorsys import hsv_to_rgb, rgb_to_hsv
import logging
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from suite2p.io import BinaryFile
from toolbox.utils.duplicates import generate_movie_preview


logger = logging.getLogger()


def preview_binary_movie(
    ops,
    movie_type,
):
    """
    Create a movie preview for a suite2p binary file
    """

    fname = "movie_preview.mp4"

    L = ops["nframes"]
    if ops["frames_include"] != -1 and ops["frames_include"] != L:
        L = ops["frames_include"]
    fs = ops["fs"]

    bin_info_dict = {
        "movie_shape": (L, ops["Ly"], ops["Lx"]),
        "movie_sampling_period": 1 / fs,
        "movie_fs": fs,
    }

    generate_movie_preview(
        movie_filename=ops[movie_type],
        preview_filename=fname,
        bin_info_dict=bin_info_dict,
    )


def preview_registration_fovs(
    ops,
    vmin_perc,
    vmax_perc,
    cmap,
    show_grid,
    ticks_step,
):
    """
    Create preview figure showing FOVs related to registration/motion correction
    """

    im_dict = {
        "refImg": "Reference image for registration",
        "max_proj": "Max projection image (registered)",
        "meanImg": "Mean projection image (reg.)",
        "meanImgE": "Enhanced mean projection image (reg.)",
    }

    Ly, Lx = ops["refImg"].shape
    xticks = np.arange(0, Lx + ticks_step, ticks_step)
    yticks = np.arange(0, Ly + ticks_step, ticks_step)

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8 * Ly / Lx), sharex=True, sharey=True
    )
    for idx, (ax, key) in enumerate(zip(axes.ravel(), list(im_dict.keys()))):
        vmin = np.percentile(ops[key], vmin_perc)
        vmax = np.percentile(ops[key], vmax_perc)
        if key == "max_proj":
            extent = (*ops["xrange"], *ops["yrange"][::-1])
        else:
            extent = None
        ax.imshow(
            ops[key],
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )
        ax.set_xticks(xticks - 0.5, xticks)
        ax.set_yticks(yticks - 0.5, yticks)
        if show_grid:
            ax.grid(ls="--", color="k")
        ax.set_title(im_dict[key])
        if idx == 2:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        ax.set_xlim((-0.5, Lx - 0.5))
        ax.set_ylim((Ly - 0.5, -0.5))
    fig.tight_layout()
    plt.savefig("registration_fovs.png")


def preview_registration_offsets(ops):
    """
    Create preview figure showing x and y offsets for rigid and non-rigid registration, as well as time course of the frame-to-reference-image phase correlation peak
    """

    arr_dict = {
        "xoff": "Rigid x offsets",
        "yoff": "Rigid y offsets",
        "xoff1": "Non-rigid x offsets",
        "yoff1": "Non-rigid y offsets",
        "corrXY": "Frame-to-reference-image phase correlation peak",
    }

    if "xoff1" not in ops:
        arr_dict.pop("xoff1")
        arr_dict.pop("yoff1")

    L = len(ops["xoff"])
    if ops["frames_include"] != -1 and ops["frames_include"] != L:
        L = ops["frames_include"]
    tb = np.arange(L) / ops["fs"]

    n_subplots = len(arr_dict)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(8, n_subplots + 3))
    for idx, (ax, key) in enumerate(zip(axes.ravel(), list(arr_dict.keys()))):
        ax.plot(tb, ops[key], lw=1)
        ax.set_title(
            f"{arr_dict[key]} (min {ops[key].min():.3g}, max {ops[key].max():.3g})"
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(tb[[0, -1]])
        if idx < 4:
            ax.set_xticklabels([])
            ax.set_ylabel("offset (px)")
        if idx == 4:
            ax.set_xlabel("time (s)")
            ax.set_ylabel("amplitude (A.U.)")
    fig.tight_layout()
    plt.savefig("registration_offsets.png")


def preview_registration_movies(
    ops,
    vmin_perc,
    vmax_perc,
    cmap,
    show_grid,
    ticks_step,
    display_rate,
    txt_padding=10,
):
    """
    Create preview movie showing raw and registered movies side-by-side
    """

    fname = "registration_movies.mp4"

    Ly, Lx = ops["refImg"].shape
    xticks = np.arange(0, Lx + ticks_step, ticks_step)
    yticks = np.arange(0, Ly + ticks_step, ticks_step)

    L = ops["nframes"]
    if ops["frames_include"] != -1 and ops["frames_include"] != L:
        L = ops["frames_include"]
    fs = ops["fs"]
    arr_frames = np.arange(0, L, display_rate)

    f_raw = BinaryFile(Ly=Ly, Lx=Lx, filename=ops["raw_file"])
    f_reg = BinaryFile(Ly=Ly, Lx=Lx, filename=ops["reg_file"])

    vmin = np.percentile(f_raw[0, :, :], vmin_perc)
    vmax = np.percentile(f_raw[0, :, :], vmax_perc)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im_raw = axes[0].imshow(f_raw[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_xticks(xticks - 0.5, xticks)
    axes[0].set_yticks(yticks - 0.5, yticks)
    if show_grid:
        axes[0].grid(ls="--", color="k")
    axes[0].set_title("raw movie")
    axes[0].set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")
    axes[0].set_xlim((-0.5, Lx - 0.5))
    axes[0].set_ylim((Ly - 0.5, -0.5))
    txt = axes[0].text(
        x=txt_padding, y=txt_padding, s="t=0s", color="w", ha="left", va="top"
    )
    im_reg = axes[1].imshow(f_reg[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_xticks(xticks - 0.5, xticks)
    axes[1].set_yticks(yticks - 0.5, yticks)
    if show_grid:
        axes[1].grid(ls="--", color="k")
    axes[1].set_title("registered movie")
    axes[1].set_xlabel("x (px)")
    axes[1].set_yticklabels([])
    axes[1].set_xlim((-0.5, Lx - 0.5))
    axes[1].set_ylim((Ly - 0.5, -0.5))
    fig.tight_layout()

    def plot_frame(idx_frame):
        im_raw.set_data(f_raw[idx_frame, :, :])
        im_reg.set_data(f_reg[idx_frame, :, :])
        txt.set_text(f"t={int(idx_frame / display_rate)}s")

    a = animation.FuncAnimation(
        fig,
        plot_frame,
        frames=arr_frames,
        interval=(1000 / int(fs)),
        repeat=False,
        blit=False,
    )

    plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
    writer = animation.FFMpegWriter(
        fps=int(fs), extra_args=["-vcodec", "libx264"]
    )
    a.save(fname, writer=writer)


def preview_detection_footprints(
    ops,
    vmin_perc,
    vmax_perc,
    cmap,
    show_grid,
    ticks_step,
):
    """
    Create preview figure showing FOV with footprints of accepted and rejected ROIs
    """

    npy_dir = ops["save_path"]
    Ly, Lx = ops["refImg"].shape

    iscell_path = f"{npy_dir}/iscell.npy"
    if os.path.exists(iscell_path):
        has_iscell = True

        stat_path = f"{npy_dir}/stat.npy"
        stat = np.load(stat_path, allow_pickle=True)
        N = len(stat)
        iscell = np.load(f"{npy_dir}/iscell.npy", allow_pickle=True)

        iscell = iscell[:, 0]
        im_idx = -1
        standalone_figure_name = "detection_footprints_accepted.png"
    else:
        has_iscell = False

        stat_path = f"{npy_dir}/stat_ROI_detection.npy"
        stat = np.load(stat_path, allow_pickle=True)
        N = len(stat)

        iscell = np.ones((N,))
        im_idx = 2
        standalone_figure_name = "detection_footprints_detected.png"

    hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)
    for idx in range(N):
        ypix = stat[idx]["ypix"]
        xpix = stat[idx]["xpix"]
        coefs = stat[idx]["lam"] / stat[idx]["lam"].max()
        hsvs[int(iscell[idx]), ypix, xpix, 0] = np.random.rand(1)
        hsvs[int(iscell[idx]), ypix, xpix, 1] = 1
        hsvs[int(iscell[idx]), ypix, xpix, 2] = coefs
    rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(
        hsvs.shape
    )

    im_list = [
        (ops["max_proj"], "Max projection image (registered)"),
        (ops["Vcorr"], "Local correlation image (registered)"),
    ]
    if has_iscell:
        im_list += [
            (rgbs[0], "Non-cell ROIs"),
            (rgbs[1], "Cell ROIs"),
        ]
    else:
        im_list += [
            (rgbs[1], "Detected ROIs"),
            (None, ""),
        ]

    xticks = np.arange(0, Lx + ticks_step, ticks_step)
    yticks = np.arange(0, Ly + ticks_step, ticks_step)

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8 * Ly / Lx), sharex=True, sharey=True
    )
    for idx, (ax, (im, title_str)) in enumerate(zip(axes.ravel(), im_list)):
        if im is None:
            ax.remove()
            continue
        if idx < 2:
            vmin = np.percentile(im, vmin_perc)
            vmax = np.percentile(im, vmax_perc)
            extent = (*ops["xrange"], *ops["yrange"][::-1])
            ax.imshow(
                im,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            grid_color = "k"
        else:
            ax.imshow(im, aspect="auto")
            grid_color = "w"
        if idx == 2:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
        ax.set_xticks(xticks - 0.5, xticks)
        ax.set_yticks(yticks - 0.5, yticks)
        if show_grid:
            ax.grid(ls="--", color=grid_color)
        ax.set_title(title_str)
        ax.set_xlim((-0.5, Lx - 0.5))
        ax.set_ylim((Ly - 0.5, -0.5))
    fig.tight_layout()
    plt.savefig("detection_footprints_all.png")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * Ly / Lx))
    im, title_str = im_list[im_idx]
    ax.imshow(im, aspect="auto")
    ax.set_xticks(xticks - 0.5, xticks)
    ax.set_yticks(yticks - 0.5, yticks)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(title_str)
    if show_grid:
        ax.grid(ls="--", color=grid_color)
    ax.set_xlim((-0.5, Lx - 0.5))
    ax.set_ylim((Ly - 0.5, -0.5))
    fig.tight_layout()
    plt.savefig(standalone_figure_name)


def preview_extraction_traces(
    ops,
    n_samp_cells,
    thresh_spks_perc,
    random_seed,
    show_all_footprints,
    show_grid,
    ticks_step,
):
    """
    Create preview figure showing randomly sampled cell fluorescence traces and underlying deconvolved spikes
    """
    np.random.seed(random_seed)

    npy_dir = ops["save_path"]
    F = np.load(f"{npy_dir}/F.npy", allow_pickle=True)
    Fneu = np.load(f"{npy_dir}/Fneu.npy", allow_pickle=True)

    stat_path = f"{npy_dir}/stat.npy"
    if os.path.exists(stat_path):
        stat = np.load(stat_path, allow_pickle=True)
        has_stat = True
    else:
        has_stat = False
    spks_path = f"{npy_dir}/spks.npy"
    if os.path.exists(spks_path):
        spks = np.load(spks_path, allow_pickle=True)
        has_spks = True
    else:
        has_spks = False
    iscell_path = f"{npy_dir}/iscell.npy"
    if os.path.exists(iscell_path):
        iscell = np.load(iscell_path, allow_pickle=True)
        has_iscell = True
    else:
        has_iscell = False

    N, L = F.shape
    if ops["frames_include"] != -1 and ops["frames_include"] != L:
        L = ops["frames_include"]
        F = F[:, :L]
        Fneu = Fneu[:, :L]
        if has_spks:
            spks = spks[:, :L]
    fs = ops["fs"]
    tb = np.arange(L) / fs
    Ly, Lx = ops["Ly"], ops["Lx"]
    figsize = (10, 2 + n_samp_cells / 2)

    if has_iscell:
        idx_iscell = np.where(iscell[:, 0] == 1)[0]
    else:
        idx_iscell = np.arange(N)
    if n_samp_cells > len(idx_iscell):
        logger.info(
            f"`Number of Sample Cells`={n_samp_cells} exceeds the number of accepted cells={len(idx_iscell)}. Reducing it to {len(idx_iscell)}.`"
        )
        n_samp_cells = len(idx_iscell)
    idx_cells = np.sort(
        np.random.choice(idx_iscell, n_samp_cells, replace=False)
    )

    # our style
    if has_spks:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for offset, idx in enumerate(idx_cells[::-1]):
            y = F[idx, :]
            y = (y - np.mean(y)) / np.max(y)
            s = spks[idx, :] > np.percentile(spks, thresh_spks_perc)
            ax.plot(tb, y + offset + 1, lw=0.5)
            ax.scatter(
                tb[s],
                np.zeros((np.sum(s),)) + np.min(y) + offset + 0.975,
                s=(spks[idx, s] / spks.max()) * 10,
                c="k",
            )
        ax.set_xlim(tb[[0, -1]])
        ax.set_xlabel("time (s)")  # , size=label_size)
        ax.set_ylim((0.5, n_samp_cells + 1))
        ax.set_ylabel("cell (index)")  # , size=label_size)
        ax.set_yticks(np.arange(1, n_samp_cells + 1), labels=idx_cells[::-1])
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(axis="both")  # , labelsize=tick_label_size)
        fig.tight_layout()
        plt.savefig("extracted_sample_sources_traces_spikes.png")
        line_colors = [x.get_color() for x in ax.lines[::-1]]

    # our style - traces only for cellset_raw
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for offset, idx in enumerate(idx_cells[::-1]):
        y = F[idx, :]
        y = (y - np.mean(y)) / np.max(y)
        ax.plot(tb, y + offset + 1, lw=0.5)
    ax.set_xlim(tb[[0, -1]])
    ax.set_xlabel("time (s)")  # , size=label_size)
    ax.set_ylim((0.5, n_samp_cells + 1))
    ax.set_ylabel("cell (index)")  # , size=label_size)
    ax.set_yticks(np.arange(1, n_samp_cells + 1), labels=idx_cells[::-1])
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both")  # , labelsize=tick_label_size)
    fig.tight_layout()
    plt.savefig("extracted_sample_sources_traces_only.png")
    line_colors = [x.get_color() for x in ax.lines[::-1]]

    # FOV with colored footprints from sample traces
    if has_stat:
        colors = ["#ffffff"] * N
        for idx, line_color in zip(idx_cells, line_colors):
            colors[idx] = line_color

        hsvs = np.zeros((Ly, Lx, 3), dtype=np.float32)
        meds = []
        for idx, color_hex in enumerate(colors):
            ypix = stat[idx]["ypix"]
            xpix = stat[idx]["xpix"]
            coefs = stat[idx]["lam"] / stat[idx]["lam"].max()
            if idx in idx_cells:
                meds.append(stat[idx]["med"])
            else:
                if not show_all_footprints:
                    continue
            color_rgb = hex_to_rgb(color_hex[1:])
            color_hsv = rgb_to_hsv(*color_rgb)
            hsvs[ypix, xpix, 0] = color_hsv[0]
            hsvs[ypix, xpix, 1] = color_hsv[1]
            hsvs[ypix, xpix, 2] = coefs  # color_hsv[2]/255
        im = np.array(
            [hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]
        ).reshape(hsvs.shape)
        xticks = np.arange(0, Lx + ticks_step, ticks_step)
        yticks = np.arange(0, Ly + ticks_step, ticks_step)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6 * Ly / Lx))
        ax.imshow(im, aspect="auto")
        ax.set_xticks(xticks - 0.5, xticks)
        ax.set_yticks(yticks - 0.5, yticks)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        for idx, med in zip(idx_cells, meds):
            ax.text(x=med[1], y=med[0], s=idx, color="w", fontsize=8)
        if show_grid:
            ax.grid(ls="--", color="w")
        ax.set_xlim((-0.5, Lx - 0.5))
        ax.set_ylim((Ly - 0.5, -0.5))
        fig.tight_layout()
        plt.savefig("extracted_sample_sources_footprints.png")

    # suite2p style
    if has_spks:
        fig, axes = plt.subplots(n_samp_cells, 1, figsize=figsize)
        for offset, idx in enumerate(idx_cells):
            ax = axes[offset]
            f = F[idx, :]
            f_neu = Fneu[idx, :]
            sp = spks[idx, :]

            fmax = np.maximum(f.max(), f_neu.max())
            fmin = np.minimum(f.min(), f_neu.min())
            frange = fmax - fmin
            sp /= sp.max()
            sp *= frange

            ax.plot(tb, f, label="Cell fluorescence", lw=1)
            ax.plot(tb, f_neu, label="Neuropil fluorescence", lw=1)
            ax.plot(tb, sp + fmin, label="Deconvolved spikes", lw=1)
            ax.set_ylabel(f"ROI {idx}")
            ax.set_xlim(tb[[0, -1]])
            ax.set_ylim((0, ax.get_ylim()[1]))
            ax.spines[["top", "right"]].set_visible(False)
            if offset == 0:
                ax.legend(loc="right")
            if offset < n_samp_cells:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("frame")
        fig.tight_layout()
        plt.savefig("extracted_sample_sources_traces_suite2p.png")

    # raster plot
    if has_spks:
        figsize_raster = (figsize[0], np.max((6, len(idx_iscell) / 20)))
        fig, ax = plt.subplots(1, 1, figsize=figsize_raster)
        ax.imshow(
            spks[idx_iscell, :],
            aspect="auto",
            cmap="Greys",
            interpolation="None",
            extent=(tb[0], tb[-1], N, 0),
            vmin=0,
            vmax=np.percentile(spks[idx_iscell, :], thresh_spks_perc),
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("cell (index)")
        fig.tight_layout()
        plt.savefig("raster_deconvolved_spikes.png")


def create_output_previews(
    ops,
    steps="all",
    vmin_perc=0,
    vmax_perc=99,
    cmap="plasma",
    show_grid=True,
    ticks_step=128,
    display_rate=10,
    n_samp_cells=20,
    thresh_spks_perc=99.7,
    random_seed=0,
    show_all_footprints=True,
):
    """
    Wrapper to create all output previews in one function call
    """
    vmin_perc, vmax_perc, ticks_step = curate_visualization_parameters(
        ops=ops,
        vmin_perc=vmin_perc,
        vmax_perc=vmax_perc,
        ticks_step=ticks_step,
    )

    if steps == "binary_conversion":
        preview_binary_movie(
            ops=ops,
            movie_type="raw_file",
        )

    if steps == "registration":
        preview_binary_movie(
            ops=ops,
            movie_type="reg_file",
        )

    if steps in ["all", "registration", "output_conversion"]:
        preview_registration_fovs(
            ops=ops,
            vmin_perc=vmin_perc,
            vmax_perc=vmax_perc,
            cmap=cmap,
            show_grid=show_grid,
            ticks_step=ticks_step,
        )

        preview_registration_offsets(ops=ops)

    if steps in ["all", "registration"]:
        preview_registration_movies(
            ops=ops,
            vmin_perc=vmin_perc,
            vmax_perc=vmax_perc,
            cmap=cmap,
            show_grid=show_grid,
            ticks_step=ticks_step,
            display_rate=display_rate,
        )

    if steps in [
        "all",
        "roi_detection",
        "roi_classification",
        "output_conversion",
    ]:
        preview_detection_footprints(
            ops=ops,
            vmin_perc=vmin_perc,
            vmax_perc=vmax_perc,
            cmap=cmap,
            show_grid=show_grid,
            ticks_step=ticks_step,
        )

    if steps in [
        "all",
        "roi_extraction",
        "spike_deconvolution",
        "output_conversion",
    ]:
        preview_extraction_traces(
            ops=ops,
            n_samp_cells=n_samp_cells,
            thresh_spks_perc=thresh_spks_perc,
            random_seed=random_seed,
            show_all_footprints=show_all_footprints,
            show_grid=show_grid,
            ticks_step=ticks_step,
        )


def hex_to_rgb(hex):
    """
    Convert HEX color to RGB color
    """
    return tuple(int(hex[idx : idx + 2], 16) for idx in (0, 2, 4))


def curate_visualization_parameters(ops, vmin_perc, vmax_perc, ticks_step):
    """
    Handle problematic inputs for the visualization parameters
    """
    if vmin_perc > vmax_perc:
        logger.info(
            f"`FOV vmin Percentile`={vmin_perc} should be smaller than `FOV vmax Percentile`={vmax_perc}. Setting them to default (0 and 99, respectively).`"
        )
        vmin_perc, vmax_perc = 0, 99

    min_L = np.min((ops["Lx"], ops["Ly"]))
    if ticks_step < min_L / 10:
        logger.info(
            f"`FOV Ticks Step`={ticks_step} is too small for proper display. It should be >{int(min_L / 10)}. Setting it to default (128).`"
        )
        ticks_step = 128
    if ticks_step > min_L:
        logger.info(
            f"`FOV Ticks Step`={ticks_step} is too large for proper display. It should be <={min_L}. Setting it to default (128).`"
        )
        ticks_step = 128

    return vmin_perc, vmax_perc, ticks_step
