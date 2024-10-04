import cv2
from enum import Enum, unique
import imageio.v2 as iio
import isx
import logging
import math
import numpy as np
import os
from suite2p.io import BinaryFile

logger = logging.getLogger()


def _transform_movie_to_preview_shape(
    movie_frame_shape: tuple[int, int], preview_frame_shape: tuple[int, int]
) -> tuple[int, int]:
    """Transform movie frame shape to optimally fit within the corresponding preview of the movie.
    Given a desired preview frame shape, determine the largest scaled version of the movie frame shape
    that fits within the preview frame shape, and maintains aspect ratio of the movie frame shape.
    At least one side of the output scaled movie frame shape will be equal to the corresponding
    side in the preview frame shape.

    :param movie_frame_shape: The shape of the movie frame. Represented as (num_rows, num_cols)
        where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
        in a movie frame.
    :param preview_frame_shape: The shape of the preview frame. Represented as (num_rows, num_cols)
        where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
        in a movie frame.

    :return scaled_frame_shape: The scaled shape of the movie frame. Represented as (num_rows, num_cols)
            where num_rows is the number of rows in a movie frame, and num_cols is the number of columns
            in a movie frame. The aspect ratio of scaled_movie_frame_shape should be very close to movie_frame_shape

    Copied from ideas-toolbox-idps/toolbox/utils/movie_preview.py
    """
    # Step 1: Check if the movie frame shape is smaller than the preview frame shape.
    # If so, return the movie frame shape.
    if np.all(
        [movie_frame_shape[i] <= preview_frame_shape[i] for i in range(2)]
    ):
        return movie_frame_shape

    # Step 2: Determine the dimension that needs to be scaled down
    # the most to equal the corresponding preview frame dimension
    # and used this as the scale factor to apply on the movie frame shape
    scale_factor = np.min(
        [preview_frame_shape[i] / movie_frame_shape[i] for i in range(2)]
    )

    # We know that at least one movie frame shape dimension is larger than the
    # corresponding preview frame shape dimension, so at least one scale factor
    # should be less than zero.
    assert scale_factor < 1

    # Step 3: Scale the movie frame shape by the scale factor
    scaled_frame_shape = np.empty(shape=(2,), dtype=int)
    for i in range(2):
        scaled_frame_shape[i] = round(
            float(movie_frame_shape[i]) * scale_factor
        )

    return tuple(scaled_frame_shape)


def _map_movie_to_preview_frame_ind(
    preview_frame_ind: int,
    preview_sampling_period: float,
    movie_num_frames: int,
    movie_sampling_period: float,
) -> list[int]:
    """Map a sequence of frames in a movie to one frame in the corresponding preview for that movie.
    Movie frames that belong to a preview frame are determined by comparing timestamps from frames in both movies.
    To demonstrate this, consider the following example:
        Movie frame rate: 13 Hz
        Preview frame rate: 10 Hz
        For 1 second of time, frames will be displayed at the following times for the movie and preview:
            Movie frame timestamps = [0.0, 0.083, 0.16, 0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75, 0.83, 0.91, 1.0]
            Preview frame timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        Let fp(i) represent the ith preview frame.
        Let fm(i) represent the ith movie frame.
        fp(0) will be a linear interpolation of all frames in the original movie that
        occurred between 0.0 s - 0.1 s:
            fp(0) = ((0.083 - 0.0) / 0.1) * fm(0) + ((0.1 - 0.083) / 0.1) * fm(1)
            fp(0) = 0.83 * fm(0) + 0.17 * fm(1)

    :param preview_frame_ind: The index of the preview frame.
        The function will determine which frames in the original movie occur during the
        duration of the preview frame.
    :param preview_sampling_period: The sampling period of the preview.
    :param movie_num_frames: The total number of frames in the original movie.
        If no frames from the original movie occur during the duration of the preview frame,
        an empty list is returned.
    :param movie_sampling_period: The sampling period of the original movie.

    :return preview_frame_map: List of tuples. Each tuple consists of a frame index and floating
        point number between 0-1 representing the fraction of time the movie frame was displayed within
        throughout the duration of the preview frame.

    Copied from ideas-toolbox-idps/toolbox/utils/movie_preview.py
    """
    # Step 1: Calculate the start and end timestamps of the preview frame
    # based on the preview sampling period and frame index.
    preview_frame_start_ts = preview_sampling_period * preview_frame_ind
    preview_frame_end_ts = preview_sampling_period * (preview_frame_ind + 1)

    # Step 2: Calculate the first movie frame that starts before the start of the preview frame.
    movie_frame_ind = math.floor(
        preview_frame_start_ts / movie_sampling_period
    )

    # Step 3: Starting from the first movie frame that starts before the start of the preview frame,
    # find all movie frames that occur during the duration of the preview frame.
    preview_frame_map = []
    while movie_frame_ind < movie_num_frames:
        # Step 3.1: Calculate the start and end timestamps of the movie frame
        # based on the movie sampling period and frame index.
        movie_frame_start_ts = movie_sampling_period * movie_frame_ind
        movie_frame_end_ts = movie_sampling_period * (movie_frame_ind + 1)

        # Step 3.2: Determine if movie frame occurs within the preview frame.
        if movie_frame_start_ts >= preview_frame_end_ts:
            # The movie frame starts after the end of the preview frame,
            # so the movie frame does not occur within the preview frame.
            # No more movie frames will occur in the preview frame, exit loop.
            break
        elif movie_frame_end_ts <= preview_frame_end_ts:
            # The movie frame ends before the preview frame,
            # so the movie frame does occur within the preview frame.
            if preview_frame_map:
                # There's already an earlier movie frame that also occurs within the preview frame.
                # Calculate when the previous movie frame ended in order to determine the current
                # movie frame contribution.
                last_frame_end_ts = movie_sampling_period * (
                    preview_frame_map[-1][0] + 1
                )
            else:
                # This is the first movie frame that occurs within the preview frame.
                # Use the start of the preview frame to determine the current movie frame contribution.
                last_frame_end_ts = preview_frame_start_ts

            # Calculate the movie frame contribution as the ratio of time that the movie frame is displayed
            # over the time that the preview frame is displayed.
            movie_frame_contribution = (
                movie_frame_end_ts - last_frame_end_ts
            ) / preview_sampling_period
            preview_frame_map.append(
                (movie_frame_ind, movie_frame_contribution)
            )

            # Move on to next frame in the movie.
            movie_frame_ind += 1
        else:
            # The movie frame ends after the preview frame,
            # so the movie frame does occur within the preview frame, but this will be the last movie frame
            # that occurs within the preview frame, exit loop after adding it to the output.

            # Calculate the movie frame contribution as the ratio of time that the movie frame is displayed
            # over the time that the preview frame is displayed.
            movie_frame_contribution = (
                preview_frame_end_ts - movie_frame_start_ts
            ) / preview_sampling_period
            preview_frame_map.append(
                (movie_frame_ind, movie_frame_contribution)
            )
            break

    # Step 4: Check edge case where sometimes the last preview frame that occurs throughout the duration
    # of the movie ends after the last movie frame. Discard this preview frame as it's incomplete.
    if movie_frame_ind == movie_num_frames and preview_frame_map:
        total_movie_frame_contribution = 0.0
        for data in preview_frame_map:
            _, movie_frame_contribution = data
            total_movie_frame_contribution += movie_frame_contribution
        if not np.allclose(total_movie_frame_contribution, 1.0):
            preview_frame_map = []

    return preview_frame_map


def generate_movie_preview(
    movie_filename,
    preview_filename,
    bin_info_dict=None,
    preview_max_duration=120,
    preview_max_sampling_rate=10,
    preview_max_resolution=(640, 400),
    preview_crf=23,
    preview_max_size=50,
):
    """
    Generate a preview for a movie.

    Copied from ideas-toolbox-idps/toolbox/utils/movie_preview.py
    Modified to accommodate .bin movies
    """
    # Step 1: Determine file type based on extension
    # Read in properties of movie in order to determine
    # preview duration, sampling rate, and resolution
    _, file_extension = os.path.splitext(movie_filename.lower())
    if file_extension in [".isxd", ".isxb"]:
        # Use isx API to read isxd and isxb movies
        movie = isx.Movie.read(movie_filename)
        frame_height, frame_width = movie.spacing.num_pixels
        num_frames = movie.timing.num_samples
        movie_sampling_period = movie.timing.period.secs_float
        movie_sampling_rate = 1 / movie_sampling_period
    elif file_extension in [".bin"]:
        if bin_info_dict is None:
            raise ToolException(
                ExitStatus.IDPS_ERROR_0002,
                "For .bin movies, you need to provide `bin_info_dict`",
            )
        num_frames, frame_height, frame_width = bin_info_dict["movie_shape"]
        movie = BinaryFile(
            Ly=frame_height,
            Lx=frame_width,
            filename=movie_filename,
        )
        movie_sampling_period = bin_info_dict["movie_sampling_period"]
        movie_sampling_rate = bin_info_dict["movie_fs"]
    else:
        raise ToolException(
            ExitStatus.IDPS_ERROR_0002,
            "Only isxd, isxb, and bin movies are supported",
        )

    # Step 2: Determine preview duration, sampling rate, and resolution
    # based on movie properties.
    preview_sampling_rate = preview_max_sampling_rate
    if preview_max_sampling_rate > movie_sampling_rate:
        # Set preview sampling rate to movie sampling rate if it's
        # less than max preview sampling rate
        preview_sampling_rate = movie_sampling_rate
    preview_sampling_period = 1 / preview_sampling_rate

    movie_duration = num_frames * movie_sampling_period
    # Preview duration is either max duration or movie duration
    preview_duration = min(preview_max_duration * 60, movie_duration)
    preview_num_frames = math.ceil(preview_sampling_rate * preview_duration)
    # Maximum bit rate for previews in units of Kb/s.
    # This is passed to ffmpeg to ensure all preview file are within a max size limit.
    # The max file size determines the max bit rate.
    preview_max_bit_rate = int((preview_max_size * 1e3 * 8) / preview_duration)

    # Frame shape is represented as (num_rows, num_cols)
    # Resolution is represented as (width, height)
    # num_rows = height and num_cols = width
    # So flip dimensions of resolution to get frame shape
    preview_frame_shape = (
        preview_max_resolution[1],
        preview_max_resolution[0],
    )
    scaled_frame_shape = _transform_movie_to_preview_shape(
        movie_frame_shape=(frame_height, frame_width),
        preview_frame_shape=preview_frame_shape,
    )

    # Step 3: Initialize video writer for preview file
    # Use imageio to write compressed file using H.264 standard
    # https://imageio.readthedocs.io/en/v2.10.0/reference/_backends/imageio.plugins.ffmpeg.html
    writer = iio.get_writer(
        preview_filename,
        format="FFMPEG",  # Use ffmpeg library to write compressed file
        mode="I",  # "I" stands for series of images to write
        fps=preview_sampling_rate,
        codec="h264",  # Use H.264 since it's currently the most widely adopted
        # video compression standard. So it should be compatible with most browsers and user devices
        output_params=[
            "-crf",
            f"{preview_crf}",
            "-maxrate",
            f"{preview_max_bit_rate}K",
            "-bufsize",
            f"{int(preview_max_bit_rate / 2)}K",
        ],
        # Parameter bufsize needs to be specified in order for the max bit rate to be set appropiately.
        # FFMPEG docs suggest that a good general value for this param is half the max bit rate.
        macro_block_size=16,  # Size constraint for video. Width and height, must be divisible by this number.
        # If not divisible by this number imageio will tell ffmpeg to scale the image up to
        # the next closest size divisible by this number.  Most codecs are compatible with a
        # macroblock size of 16 (default). Even though this is the default value for this function
        # I'm leaving this here so others are aware in the future of why the resolution of the preview
        # may not exactly match the resolution of the movie after it's been scaled down to fit within
        # `preview_max_resolution`. It is possible to use a smaller value like 4, 2, or even 1,
        # but many players are not compatible with smaller values so I didn't want to take the risk.
        ffmpeg_log_level="error",  # ffmpeg can be quite chatty with warnings.
        # Setting to error in order to avoid cluttering logs.
    )

    # Step 4: Write frames to preview file
    # Often times one movie frame will appear on the boundary of two consecutive
    # preview frames. In order to prevent reading the same movie frame more than
    # once, keep track of the last movie frame that was read for the previous
    # preview frame that was processed in the loop.
    last_movie_frame_ind = None  # Index of last movie frame that was read for the previous preview frame
    last_movie_frame = None  # Frame data of the last movie frame that was read for the previous preview frame
    for preview_frame_ind in range(preview_num_frames):
        # Step 4.1: Find movie frames that occur within the current preview frame
        preview_frame_map = _map_movie_to_preview_frame_ind(
            preview_frame_ind=preview_frame_ind,
            preview_sampling_period=preview_sampling_period,
            movie_num_frames=num_frames,
            movie_sampling_period=movie_sampling_period,
        )

        # Step 4.2: Iterate through all movie frames that occur within the preview frame
        preview_frame = None  # Initialize preview frame to empty object
        num_mapped_movie_frames = len(
            preview_frame_map
        )  # Number of movie frames that occur within
        # the preview frame
        for mapped_movie_frame_ind, mapped_movie_frame in enumerate(
            preview_frame_map
        ):
            # Step 4.2.1: Unpack data in current entry of the previw frame map.
            # Preview frame map returns a frame index, and a floating point number
            # representng the contribution that the movie frame makes to the preview frame
            movie_frame_ind, movie_frame_contribution = mapped_movie_frame

            # Step 4.2.2: Get movie frame data
            # See if first frame of the movie sequence is equal to the last frame of the last movie sequence
            if (
                mapped_movie_frame_ind == 0
                and last_movie_frame is not None
                and movie_frame_ind == last_movie_frame_ind
            ):
                # Use cached frame data instead of re-reading from movie file
                movie_frame = last_movie_frame
            else:
                # Assert that the next movie frame read into memory is the next
                # successive frame in the movie. That is, no movie frames (within a particular duration)
                # are skipped when generating previews, and movie frames are processed in order.
                # This is important for mp4 and avi files because we read the next frame
                # in the file for those formats since it's more efficient than seeking
                # before reading each frame.
                assert last_movie_frame_ind is None or movie_frame_ind == (
                    last_movie_frame_ind + 1
                )

                # Read movie frame based on file type
                if file_extension in [".isxd", ".isxb"]:
                    movie_frame = movie.get_frame_data(movie_frame_ind)
                elif file_extension in [".bin"]:
                    movie_frame = movie.data[movie_frame_ind]
                else:
                    raise ToolException(
                        ExitStatus.IDPS_ERROR_0002,
                        "Only isxd, isxb, and bin movies are supported",
                    )

                # Convert the frame to floating point in order to perform linear interpolation later
                movie_frame = movie_frame.astype(np.float64)

                # Keep track of last movie frame that was read from the file
                last_movie_frame_ind = movie_frame_ind

                # If this is the last movie frame in the sequence of movie frames that occur during
                # the preview frame, save a copy of the frame data because it will most likely
                # be used in the next preview frame as well
                if mapped_movie_frame_ind == (num_mapped_movie_frames - 1):
                    last_movie_frame = movie_frame.copy()

            # Step 4.2.3: Add the movie frame to the preview frame
            # Multiply the movie frame by the fraction of time it occurred within the preview frame
            # Preview frame is the linear interpolation of all frames that occurred durin its duration
            movie_frame *= movie_frame_contribution
            if preview_frame is None:
                # Initialize preview frame
                preview_frame = movie_frame
            else:
                # Add movie frame to existing preview frame
                preview_frame += movie_frame

        # Step 4.3: Write final preview frame to file
        if preview_frame is not None:
            # Resize the preview frame if it needs to scaled down to fit within the max resolution
            if scaled_frame_shape != (frame_height, frame_width):
                preview_frame = cv2.resize(
                    preview_frame,
                    (scaled_frame_shape[1], scaled_frame_shape[0]),
                )

            # Normalize the image and convert to 8 bit unsigned int data type
            # This data type is required for compressed video files.
            preview_frame = cv2.normalize(
                preview_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            writer.append_data(preview_frame)
        else:
            # This means no movie frames occurred throughout the duration of the preview frame
            # Exit the loop and close the preview file
            break

    writer.close()


@unique
class ExitStatusType(Enum):
    """Define exit status types."""

    SUCCESS = "SUCCESS"
    WARN = "WARN"
    ERROR = "ERROR"


@unique
class ExitStatus(Enum):
    """Define tool exit statuses.
    - No exit status is written to disk upon successful completion of a tool.
    - Error codes (ERROR) indicate unsuccessful completion and specify the type of error that occurred.
    - Warning codes (WARN) indicate deviations from normal processing conditions.
    """

    # Error codes
    IDPS_ERROR_0001 = ExitStatusType.ERROR, "File not found"
    IDPS_ERROR_0002 = ExitStatusType.ERROR, "Unsupported file type"
    IDPS_ERROR_0003 = ExitStatusType.ERROR, "Invalid file content"
    IDPS_ERROR_0004 = ExitStatusType.ERROR, "Invalid file footer metadata"
    IDPS_ERROR_0005 = ExitStatusType.ERROR, "Missing file footer metadata"
    IDPS_ERROR_0006 = (
        ExitStatusType.ERROR,
        "Invalid value provided for a tool parameter",
    )
    IDPS_ERROR_0007 = ExitStatusType.ERROR, "Missing required tool parameter"
    IDPS_ERROR_0008 = ExitStatusType.ERROR, "Missing required file"
    IDPS_ERROR_0009 = ExitStatusType.ERROR, "Invalid file metadata"
    IDPS_ERROR_0010 = ExitStatusType.ERROR, "Incompatible files"
    IDPS_ERROR_0011 = ExitStatusType.ERROR, "Incomplete results"
    IDPS_ERROR_0012 = ExitStatusType.ERROR, "Invalid output manifest record"
    IDPS_ERROR_0013 = ExitStatusType.ERROR, "No data detected"
    IDPS_ERROR_0014 = ExitStatusType.ERROR, "Invalid input combination"

    # Warning codes
    # IDPS_WARN_0001 = ExitStatusType.WARN, "Iterative algorithm reached its maximum number of iterations before convergence"
    # IDPS_WARN_0002 = ExitStatusType.WARN, "Tool parameter(s) were overridden during processing"


LOG_EXIT_STATUS_FILE = (
    "exit_status.txt"  # from ideas-toolbox-idps/toolbox/utils/config.py
)


def log_exit_status(exit_code, message):
    """Append exit status to log file."""
    with open(LOG_EXIT_STATUS_FILE, "a") as f:
        f.write("{0},{1}\n".format(exit_code.name, message))


class ToolException(Exception):
    """Exception raised for errors that occur in the tools."""

    def __init__(self, exit_code, message):
        """Initialize ToolException instance."""
        self.exit_code = exit_code
        self.message = message
        log_exit_status(exit_code, message)
        logger.error(f"{message}")
        super().__init__(self.message)
