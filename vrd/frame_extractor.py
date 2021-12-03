""" Extracts and indexes frames from a video directory
"""
import copy
import glob
import os
from pathlib import Path

import ffmpeg
import numpy as np
from tqdm.auto import tqdm


class FrameExtractor:
    """Extracts frame images from video files using ffmpeg.

    Also stores a list of the videos, as well as

    Returns:
        [type]: [description]
    """

    video_directory: Path
    frame_directory: Path
    video_extensions: list
    fps_extracted: int
    all_images: list
    fps_extracted: int
    video_list: list
    cached_video_index: dict
    verbosity: int

    def copy(self):
        """Makes a copy"""
        return copy.deepcopy(self)

    def get_index_from_video_name(self, video_name: str) -> set:
        """
        Gets the indexes matching the given video in the all_images list;
        essentially, the list extracted frames.

        Args:
            video_name (str): The video

        Returns:
            list: A set of the frames.
        """
        return set(self.cached_video_index[video_name])

    def __init__(
        self,
        video_directory,
        frame_directory,
        fps_extracted=1,
        additional_video_extensions=None,
        verbosity=0,
    ):
        """Initialize the frame extractor

        Args:
            video_directory ([type]): The directory to scan for video files
            frame_directory ([type]): Where the frame images are stored
            fps_extracted (int, optional): How many frames per second to extract, default 1 frame per second.
            additional_video_extensions ([type], optional): Additional video extensions to scan for; by default, .mp4, .avi and .flv are expected
        """
        self.video_extensions = [".mp4", ".avi", ".flv"]
        self.cached_video_index = {}
        self.fps_extracted = fps_extracted
        self.video_list = []
        self.verbosity = verbosity
        if video_directory is not Path:
            video_directory = Path(video_directory)
        assert video_directory.exists()
        assert video_directory.is_dir()

        if frame_directory is not Path:
            frame_directory = Path(frame_directory)

        if not frame_directory.exists():
            os.makedirs(frame_directory)
        assert frame_directory.exists()
        assert frame_directory.is_dir()

        self.video_directory = video_directory
        self.frame_directory = frame_directory

        if additional_video_extensions is not None:
            self.video_extensions.append(additional_video_extensions)

        self.extract_frames()

    def extract_frames(self):
        """Extract frames for all video files in the video directory specified.

        While doing so, populates the self.video_list also.
        """
        video_directory = self.video_directory
        frame_directory = self.frame_directory

        # print(f'Directory: {video_directory}')
        videos = []
        for extension in self.video_extensions:
            videos.extend(
                glob.glob(f"{video_directory}/**/*{extension}", recursive=True)
            )

        # print(f'Number of videos found: {len(videos)}')

        for vid in tqdm(videos, position=0, leave=True):

            tqdm._instances.clear()
            self._extract_png_from_video(vid, frame_directory)

        self.all_images = sorted(
            glob.glob(f"{self.frame_directory}/**/*.png", recursive=True)
        )

        # Create all indexes
        self.initialize_indexes()

        # Validate that all files produced an actual output; if not, print a warning
        # print(f'Videos in index: {len(self.video_list)}')
        # print(f'Videos in directory: {len(videos)}')
        # print(f'Unique videos: {len(np.unique(map(os.path.basename, videos)))}')

        # TODO: Do this using unique(); efforts were made but failed. This simple approach works.
        prev = None
        for vid in sorted(list(map(os.path.basename, videos))):
            if vid == prev:
                print(f"Found duplicate of file {vid}")
            prev = vid

    def initialize_indexes(self):
        """Calculates the indexes for all videos and saves them for later lookup
        Also saves the names of all videos for later use.

        """
        images = self.all_images
        filenames = list(map(self.get_video_name, images))
        vid_names, vid_indexes = np.unique(filenames, return_inverse=True)
        vid_names = sorted(vid_names)
        self.video_list = vid_names

        video_index = {}
        for i, vid in enumerate(vid_names):
            matching_index = (vid_indexes == i).nonzero()[0]
            # TODO: This should instead be (min,max) tuple but will require some refactoring!
            video_index[vid] = set(range(matching_index[0], matching_index[-1] + 1))
        self.cached_video_index = video_index

    def _extract_png_from_video(self, video_path: Path, frame_directory: Path):
        """Extracts frames from a video file, and places them in the correct output directory.



        Args:
            video_path (Path): The path to the video
            frame_directory (Path): The path to the output directory
        """
        vid_dir = self.video_directory
        subdir = os.path.relpath(video_path, vid_dir)
        filename = os.path.basename(video_path)
        out_path = Path(f"{frame_directory}//{subdir}")
        if self.verbosity > 1:
            print(f"Output path (expected): {out_path}")
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            #         print(f'Found directory for {filename}, skipping extraction...')
            return
        try:
            (
                ffmpeg.input(video_path)
                .filter("fps", fps=self.fps_extracted)
                .output(
                    f"{frame_directory}//{subdir}//{filename}_frame_%06d.png",
                    video_bitrate="5000k",
                    start_number=0,
                )
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as error:
            # TODO: Use logger instead? Throw exception?
            print("stdout:", error.stdout.decode("utf8"))
            print("stderr:", error.stderr.decode("utf8"))

    def get_video_name_from_index(self, index: int):
        """Gets the video name related to a given frame index

        Args:
            index (int): The frame indx

        Returns:
            [str]: The name of the video the frame belongs to
        """
        frame_name = self.all_images[index]
        return FrameExtractor.get_video_name(frame_name)

    def get_video_and_start_time_from_index(self, idx):
        """Gets video and start time from index"""
        frame = self.all_images[idx]
        video = self.get_video_name(frame)
        frame_start = idx - np.min(list(self.cached_video_index[video]))
        return (video, frame_start)

    def print_frame_information(self, frame):
        """Get all information from a frame path"""
        path = Path(frame)

        print(f"exists: {path.exists()}")
        print(f"suffix: {path.suffix}")
        print(f"name: {path.name}")
        print(f"suffixes: {path.suffixes}")
        print(f"is_absolute: {path.is_absolute()}")
        print(f"as_uri: {path.as_uri()}")
        print("is_relative_to: N/A")
        print(f"parent: {path.parent}")
        print("parents:")
        for par in path.parents:
            print(f"\t{par}")
        print(f"anchor: {path.anchor}")
        print(f"root: {path.root}")
        print(f"drive: {path.drive}")
        print(f"parts: {path.parts}")
        print(
            f"relative_to (frame directory): {path.relative_to(self.frame_directory)}"
        )

    def get_subdir_for_frame(self, image):
        """Gets the subdirectory of the frame (if any)

        Args:
            image (str): The path to the image

        Returns:
            str: the subdirectory, or None
        """
        return FrameExtractor.get_subdir_for_frame_static(image, self.frame_directory)

    @staticmethod
    def get_subdir_for_frame_static(image, frame_directory):
        """Gets the subdir from the specified frame.

        Args:
            image ([str]): Path to frame
            frame_directory ([str]): The base directory

        Returns:
            [Path]: The subdirectory of the frame
        """
        try:
            path_to_image = Path(image)
            rel_path = path_to_image.relative_to(frame_directory)
            return rel_path.parts[0]
        except (ValueError, TypeError):
            print("Exception!")  # Likely because there was no subdirectory...
            return None

    @staticmethod
    def get_video_name(filename: str):
        """Get the video name from the full path of a frame image

        Args:
            filename (str): The path to the frame image

        Returns:
            [type]: The name of the source video
        """
        return os.path.basename(os.path.dirname(filename))
