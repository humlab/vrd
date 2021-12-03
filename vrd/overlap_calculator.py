from copy import deepcopy
from multiprocessing import Pool

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
from matplotlib.font_manager import findSystemFonts
from numpy.lib.utils import deprecate
from PIL import ImageDraw, ImageFont

from .neighbours import Neighbours


class OverlapCalculator:
    """Finds overlaps - i.e. sequences - in the neighbours of two different frames.

    This is intended to enable finding longer stretches of matches indicative of actual matches.


    """

    video_overlap: dict
    found_sequences: dict
    neighbours: Neighbours
    videos: list

    found_two_way_sequences: dict
    two_way_overlap: dict

    def __init__(
        self,
        neighbours: Neighbours,
        maximum_distance: int,
        minimum_sequence_length=5,
        allowed_sequence_gap=0,
    ):
        """Finds overlapping sequences between video files from the precalculated distance map (neighbours).

        Args:
            neighbours (Neighbours): The neighbours list
            maximum_distance (int): Maximum allowed distance
            minimum_sequence_length (int, optional): How many frames in a row must match. Defaults to 5.
            allowed_sequence_gap (int, optional): If any gaps in the sequence is allowed, e.g. MM-MM (M = match, - = no match) would count as a sequence if this number was 1. Defaults to 0.
        """
        # TODO: Validate if this makes sense
        self.neighbours = neighbours.copy()
        self.video_overlap = {}
        self.found_sequences = {}

        self.neighbours.filter_maximum_distance(max_distance=maximum_distance)

        self.videos = list(neighbours.frames.cached_video_index.keys())

        vid_overlap = {v: self.separate_neighbours(v) for v in self.videos}
        # vid_overlap = {}
        # for v in self.videos:
        #     vid_overlap[v] = self.separate_neighbours(v)

        vid_list = self.neighbours.frames.video_list
        for video_name, (
            separated_neighbours,
            video_belonging_sorted,
        ) in vid_overlap.items():
            self.video_overlap[video_name] = {}
            self.found_sequences[video_name] = {}

            for o_vid_index, overlap in self.find_overlapping_segments(
                video_name, separated_neighbours
            ).items():
                self.video_overlap[video_name][vid_list[o_vid_index]] = overlap
                if overlap is None:
                    self.found_sequences[video_name][vid_list[o_vid_index]] = None
                else:
                    sequences = OverlapCalculator.find_sequences(
                        overlap, minimum_sequence_length, allowed_sequence_gap
                    )
                    self.found_sequences[video_name][vid_list[o_vid_index]] = sequences

        # Add two-way overlaps and sequences
        two_way_overlap = OverlapCalculator.two_way_overlaps(
            self.video_overlap, self.videos
        )
        found_two_way_sequences = {}

        for vid_1 in self.videos:
            found_two_way_sequences[vid_1] = {}
            for vid_2 in self.videos:
                tw_overlap = two_way_overlap[vid_1][vid_2]
                if tw_overlap is None:
                    found_two_way_sequences[vid_1][vid_2] = None
                else:
                    found_two_way_sequences[vid_1][
                        vid_2
                    ] = OverlapCalculator.find_sequences(
                        tw_overlap, minimum_sequence_length, allowed_sequence_gap
                    )
        self.found_two_way_sequences = found_two_way_sequences
        self.two_way_overlap = two_way_overlap

    @staticmethod
    def two_way_overlaps(video_overlap, videos):
        """Only allows two-way overlaps, i.e. that frame 1 considers frame 2 a neighbour and the other way around.

        Args:
            video_overlap ([type]): [description]
            videos ([type]): [description]

        Returns:
            [type]: [description]
        """
        result = {video: dict() for video in videos}

        for vid1 in videos:

            for vid2 in videos:
                if vid1 == vid2:
                    result[vid1][vid2] = None
                    continue
                v1v2 = video_overlap[vid1][vid2]
                v2v1 = video_overlap[vid2][vid1]
                try:
                    # TODO: Verify if this should AND or OR: Currently we run OR. The reason being that we expect the distance a->b to be equal to b->a !
                    two_way_and = np.logical_or(v1v2, v2v1.T)
                    result[vid1][vid2] = two_way_and
                    result[vid2][vid1] = two_way_and.T
                except:
                    result[vid1][vid2] = None
                    result[vid2][vid1] = None
        return result

    @staticmethod
    def either_way_overlaps(video_overlap, videos):
        """Allows either way overlaps, i.e. that frame 1 considers frame 2 a neighbour OR the other way around.

        Args:
            video_overlap ([type]): The video overlap as calculated by
            videos ([type]): [description]

        Returns:
            [type]: [description]
        """
        result = {video: dict() for video in videos}

        for vid1 in videos:

            for vid2 in videos:
                if vid1 == vid2:
                    result[vid1][vid2] = None
                    continue
                v1v2 = video_overlap[vid1][vid2]
                v2v1 = video_overlap[vid2][vid1]
                try:
                    two_way_and = np.logical_or(v1v2, v2v1.T)
                    result[vid1][vid2] = two_way_and
                    result[vid2][vid1] = two_way_and.T
                except:
                    result[vid1][vid2] = None
                    result[vid2][vid1] = None
        return result

    def separate_neighbours(self, vid: str):
        frames = self.neighbours.frames
        total_video_count = len(frames.video_list)
        vid_indexes = self.neighbours.frames.get_index_from_video_name(vid)

        # Get all neighbour indexes belonging to the specified video, and sort it
        matching_indexes = [
            (_i[0], sorted(_i[1:]))
            for _d, _i in self.neighbours.distance_list
            if _i[0] in vid_indexes and len(_i) > 1
        ]

        video_to_index = {vid: i for i, vid in enumerate(frames.video_list)}

        # Precalculate which frames belong to which video; sort by starting frame, and save start:stop value as tuple with the video name as dict key.
        video_belonging_sorted = dict(
            sorted(
                {
                    video_to_index[key]: (min(value), max(value))
                    for key, value in frames.cached_video_index.items()
                }.items(),
                key=lambda keyval: keyval[1][0],
            )
        )
        #     print(video_belonging_sorted)
        # Create result array
        vid_no_frames = len(vid_indexes)
        vid_start_frame = min(vid_indexes)
        # result_array = np.zeros((vid_frames,len(thumb.video_list)), dtype=np.bool)

        # TODO: Check if we can move this and maintain performance, consider slicing (and starting iteration at a later index)!
        def get_matching_video_index(index: int):
            for vid, (_min, _max) in video_belonging_sorted.items():
                if index >= _min and index <= _max:
                    return vid

        # Create the list-of-lists with results for easier check
        # The list is list[video_frame][video_id]
        result_list = [
            [set() for y in range(total_video_count)] for x in range(vid_no_frames)
        ]

        # Note: We assume that high-distance matches have already been filtered, and all remaining should be counted as equals.
        for frame, match_frame in matching_indexes:
            # frame is frame for current video, match_frame is all others
            result_index = frame - vid_start_frame
            # TODO: CONSIDER THAT THE MATCH FRAME LIST IS SORTED TO IMPROVE PERFORMANCE
            for single_match in match_frame:
                result_list[result_index][get_matching_video_index(single_match)].add(
                    single_match
                )

        # The list-of-lists, one list per frame containing one list per other video.
        return result_list, video_belonging_sorted

    def find_overlapping_segments(self, vid: str, separate_neighbours):
        vid_frame_count = len(self.neighbours.frames.get_index_from_video_name(vid))

        overlap_dict = {}
        for o_vid_index, o_vid_name in enumerate(self.neighbours.frames.video_list):
            if o_vid_name == vid:
                # This can't possibly be useful otherwise right?
                overlap_dict[o_vid_index] = None
                continue
            o_vid_frames = self.neighbours.frames.get_index_from_video_name(o_vid_name)
            o_vid_start = min(o_vid_frames)
            o_vid_frame_count = len(o_vid_frames)

            match_matrix = np.zeros((vid_frame_count, o_vid_frame_count), dtype=bool)
            for _i, vid_frame in enumerate(separate_neighbours):
                o_vid_matches = vid_frame[o_vid_index]
                if len(o_vid_matches) == 0:
                    continue
                #             print(o_vid_matches)
                # Set values in result matrix to true for all matches
                o_vid_matches_indexes = [x - o_vid_start for x in o_vid_matches]
                #             print(f'Video index {o_vid_index} Setting matches, frame {_i}, total {len(o_vid_matches)} matches')
                match_matrix[_i, o_vid_matches_indexes] = True
            overlap_dict[o_vid_index] = match_matrix
        #         return match_matrix
        return overlap_dict

    @deprecate
    def show_images_starting_at(
        self, vid1_index, vid2_index, vid1_start, vid2_start, window_length
    ):
        """
        Shows a sequence of images from two videos.
        Each has a separate starting point, and the sequence continues for the set amount of frames.

        Largely superseded by create_sequence_comparison_image

        """
        frames = self.neighbours.frames
        vid1name = frames.video_list[vid1_index]
        vid2name = frames.video_list[vid2_index]

        vid1_startframe = list(frames.get_index_from_video_name(vid1name))[vid1_start]
        vid2_startframe = list(frames.get_index_from_video_name(vid2name))[vid2_start]

        for i in range(window_length):
            vid1_frame = frames.all_images[vid1_startframe + i]
            vid2_frame = frames.all_images[vid2_startframe + i]
            print(f"Left:  {vid1_frame}\nRight: {vid2_frame}")

            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.imshow(mpimg.imread(vid1_frame))
            ax = fig.add_subplot(122)
            ax.imshow(mpimg.imread(vid2_frame))
            plt.show()

    def _video_frame_to_global_index(self, video, frame):
        return min(self.neighbours.frames.cached_video_index[video]) + frame

    def create_sequence_comparison_image(
        self,
        save_path,
        vid1_index,
        vid2_index,
        vid1_start,
        vid2_start,
        window_length,
        image_resize=(600, 600),
        font_size=16,
    ):
        """TODO: Fix this description; Consider the save_path

        Args:
            save_path ([type]): [description]
            vid1_index ([type]): [description]
            vid2_index ([type]): [description]
            vid1_start ([type]): [description]
            vid2_start ([type]): [description]
            window_length ([type]): [description]
            image_resize (tuple, optional): [description]. Defaults to (600,600).
            font_size (int, optional): [description]. Defaults to 24.

        Returns:
            [type]: [description]
        """
        frames = self.neighbours.frames
        vid1name = frames.video_list[vid1_index]
        vid2name = frames.video_list[vid2_index]

        vid1_startframe = self._video_frame_to_global_index(vid1name, vid1_start)
        vid2_startframe = self._video_frame_to_global_index(vid2name, vid2_start)

        # print(f'Video 1 start frame: {vid1_startframe}')
        # print(f'Video 2 start frame: {vid2_startframe}')

        font = self._get_font(font_size)

        merged = PIL.Image.new(
            "RGB",
            (image_resize[0] * 2, image_resize[1] * window_length),
            (250, 250, 250),
        )

        for i in range(window_length):
            vid1_frame = frames.all_images[vid1_startframe + i]
            vid2_frame = frames.all_images[vid2_startframe + i]
            im1 = PIL.Image.open(vid1_frame).resize(image_resize)
            im2 = PIL.Image.open(vid2_frame).resize(image_resize)

            draw_im1 = ImageDraw.Draw(im1)
            draw_im2 = ImageDraw.Draw(im2)

            vid1_distance, vid2_distance = self._find_distance_between_two_frames(
                vid1_startframe + i, vid2_startframe + i
            )

            vid1_text = f"{vid1name}:{vid1_startframe+i}\nDistance: {vid1_distance}"
            vid2_text = f"{vid2name}:{vid2_startframe+i}\nDistance: {vid2_distance}"

            draw_im1.text((0, 0), vid1_text, (255, 255, 255), font=font)
            draw_im2.text((0, 0), vid2_text, (255, 255, 255), font=font)

            merged.paste(im1, (0, i * image_resize[1]))
            merged.paste(im2, (image_resize[0], i * image_resize[1]))

        return merged

    def _find_distance_between_two_frames(self, f1, f2):
        """Finds the distance between two frames.

        TODO: Verify that they are equidistant, i.e. that f1->f2 == f2->f1

        Args:
            f1 ([type]): Frame 1, as index
            f2 ([type]): Frame 2, as index

        Returns:
            [type]: The distances between them
        """
        dlist = self.neighbours.distance_list

        f1_d, f1_i = next(iter([(d, i) for d, i in dlist if i[0] == f1]), (None, None))
        f2_d, f2_i = next(iter([(d, i) for d, i in dlist if i[0] == f2]), (None, None))

        # Consider if making the distance negative makes sense if it isn't found...
        if f1_d is None or f2_d is None:
            print(f"Could not find {f1} or {f2}")
            return (-1, -1)

        f1_dist = next(iter([d for d, i in zip(f1_d, f1_i) if i == f2]), -1)
        f2_dist = next(iter([d for d, i in zip(f1_d, f1_i) if i == f2]), -1)

        return (f1_dist, f2_dist)

    def create_gif(
        self,
        gif_path,
        vid1_index,
        vid2_index,
        vid1_start,
        vid2_start,
        window_length,
        image_resize=(600, 600),
    ):
        """Creates a gif file from the specified sequence.

        Args:
            gif_path ([type]): Where to save the gif
            vid1_index ([type]): The video index of the first video
            vid2_index ([type]): The video index of the second video
            vid1_start ([type]): Video 1 start time
            vid2_start ([type]): Video 2 start time
            window_length ([type]): How many frames the gif is
            image_resize (tuple, optional): The resized size of each extracted frame. Defaults to (600,600).
        """
        frames = self.neighbours.frames
        vid1name = frames.video_list[vid1_index]
        vid2name = frames.video_list[vid2_index]

        vid1_startframe = list(frames.get_index_from_video_name(vid1name))[vid1_start]
        vid2_startframe = list(frames.get_index_from_video_name(vid2name))[vid2_start]

        image_list = []

        for i in range(window_length):
            vid1_frame = frames.all_images[vid1_startframe + i]
            vid2_frame = frames.all_images[vid2_startframe + i]
            im1 = PIL.Image.open(vid1_frame).resize(image_resize)
            im2 = PIL.Image.open(vid2_frame).resize(image_resize)
            merged = PIL.Image.new(
                "RGB", (image_resize[0] * 2, image_resize[1]), (250, 250, 250)
            )
            merged.paste(im1, (0, 0))
            merged.paste(im2, (image_resize[0], 0))
            image_list.append(merged)
        image_list[0].save(
            fp=gif_path,
            format="GIF",
            append_images=image_list,
            save_all=True,
            duration=1000,
            loop=0,
        )  # TODO: Fix duration / fps? Currently assumes 1 image per 1000ms

    @staticmethod
    def run_gif_multiprocessing(args):
        gif_path, video_frames, image_resize = args

        image_list = []

        for vid1_frame, vid2_frame in video_frames:
            im1 = PIL.Image.open(vid1_frame).resize(image_resize)
            im2 = PIL.Image.open(vid2_frame).resize(image_resize)
            merged = PIL.Image.new(
                "RGB", (image_resize[0] * 2, image_resize[1]), (250, 250, 250)
            )
            merged.paste(im1, (0, 0))
            merged.paste(im2, (image_resize[0], 0))
            image_list.append(merged)
        image_list[0].save(
            fp=gif_path,
            format="GIF",
            append_images=image_list,
            save_all=True,
            duration=1000,
            loop=0,
        )  # TODO: Fix duration / fps? Currently assumes 1 image per 1000ms

    def create_gifs_from_overlap(self, thread_count=6):
        frames = self.neighbours.frames
        all_videos = frames.video_list

        converter = {vid: i for i, vid in enumerate(frames.video_list)}
        size_dict = {vid: len(i) for vid, i in frames.cached_video_index.items()}

        mp_parameters = []

        pool = Pool(thread_count)

        for vid1 in all_videos:
            for vid2 in all_videos:
                if vid1 == vid2:
                    continue
                v1size = size_dict[vid1]
                v2size = size_dict[vid2]

                for (diag_number, offset), length in self.found_sequences[vid1][
                    vid2
                ].items():
                    vid1_start, vid2_start = self.diag_number_to_index(
                        (v1size, v2size), diag_number, offset
                    )
                    params = self.create_gif_multiprocessing_queue(
                        f"./gifs/{vid1}:{vid1_start}_{vid2}:{vid2_start}_{length}s.gif",
                        converter[vid1],
                        converter[vid2],
                        vid1_start,
                        vid2_start,
                        length,
                    )
                    mp_parameters.append(params)

        pool.map(OverlapCalculator.run_gif_multiprocessing, mp_parameters)

    def create_gif_multiprocessing_queue(
        self,
        gif_path,
        vid1_index,
        vid2_index,
        vid1_start,
        vid2_start,
        window_length,
        image_resize=(600, 600),
    ):
        frames = self.neighbours.frames
        vid1name = frames.video_list[vid1_index]
        vid2name = frames.video_list[vid2_index]

        vid1_startframe = list(frames.get_index_from_video_name(vid1name))[vid1_start]
        vid2_startframe = list(frames.get_index_from_video_name(vid2name))[vid2_start]

        video_frames = [
            (
                frames.all_images[vid1_startframe + i],
                frames.all_images[vid2_startframe + i],
            )
            for i in range(window_length)
        ]
        return [gif_path, video_frames, image_resize]

    @staticmethod
    def _get_font(font_size):
        installed_fonts = findSystemFonts(fontpaths=None, fontext="ttf")
        # Change index here if required
        return ImageFont.truetype(installed_fonts[0], font_size)

    @staticmethod
    def get_all_diagonals(nparray):
        """Generator that returns each diagonal from a numpy array

        Args:
            nparray ([type]): The numpy array

        Yields:
            [type]: [description]
        """
        range_interval = ((nparray.shape[0] - 1) * -1, nparray.shape[1] - 1)
        for diag in range(*range_interval):
            yield np.diag(nparray, k=diag)

    @staticmethod
    def diag_number_to_index(array_size, diag_number, offset=0):
        """
        Offset is the number in the diagonal.

        Note: No verification if the number is valid is done
        """
        if diag_number < 0:
            # intersects orig_size axis
            return ((diag_number * -1) + offset, offset)

        return (offset, diag_number + offset)

    @staticmethod
    def find_sequences(match_array, minimum_length, allowed_gap_length):
        """Finds sequences, i.e. (nearly) unbroken streams of True values in the given match_array

        Args:
            match_array ([type]): A 2d numpy array of bools
            minimum_length ([type]): The minimum length to require
            allowed_gap_length ([type]): How long a gap can be in a sequence without starting over

        Returns:
            [type]: A dict of all matches
        """

        diagonals = OverlapCalculator.get_all_diagonals(match_array)
        diag_numbers = range(
            *((match_array.shape[0] - 1) * -1, match_array.shape[1] - 1)
        )

        matches = {}
        for diag_number, diagonal in zip(diag_numbers, diagonals):
            current_start = -1
            remaining_gap = allowed_gap_length
            for i, val in enumerate(diagonal):
                if val and current_start < 0:
                    current_start = i
                    remaining_gap = allowed_gap_length  # Reset this
                if not val and current_start >= 0:
                    remaining_gap -= 1
                    if remaining_gap < 0:
                        total_sequence_length = i - current_start - allowed_gap_length
                        if total_sequence_length >= minimum_length:
                            matches[
                                (diag_number, current_start)
                            ] = total_sequence_length
                        current_start = -1
            if current_start >= 0:
                total_sequence_length = i - current_start - allowed_gap_length
                if total_sequence_length >= minimum_length:
                    matches[(diag_number, current_start)] = total_sequence_length
        return matches

    def show_overlap_image(self, vid1, vid2):
        """Creates a black-and-white comparison image showing the overlap between two videos.

        This is especially useful to detect large areas of overlap, perhaps due static images in both videos.

        Args:
            vid1 ([type]): Video 1
            vid2 ([type]): Video 2
        """
        img = self.video_overlap[vid1][vid2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap="gray")

    def generate_sequence(self, vid1, vid2, two_way=True, sequence_override=None):
        """Create images of all sequences in a combination of videos

        Args:
            vid1 ([type]): Video 1
            vid2 ([type]): Video 2
            two_way (bool, optional): Whether to use two-way sequences only or not. Defaults to True.
            sequence_override: A list of sequences; mainly used if sequences are first sorted.
        """
        size_dict = {
            vid: len(i) for vid, i in self.neighbours.frames.cached_video_index.items()
        }
        converter = {vid: i for i, vid in enumerate(self.videos)}

        merged_image = None

        sequences = sequence_override
        if sequences is None:
            if two_way:
                sequences = self.found_two_way_sequences[vid1][vid2]
            else:
                sequences = self.found_sequences[vid1][vid2]

        merged_images = []

        for (diag_number, offset), length in sequences.items():
            v1size = size_dict[vid1]
            v2size = size_dict[vid2]

            vid1_start, vid2_start = self.diag_number_to_index(
                (v1size, v2size), diag_number, offset
            )
            #             oc.create_gif(f'gifs/{vid1}:{vid1_start}_{vid2}:{vid2_start}_{length}s.gif',converter[vid1], converter[vid2], vid1_start, vid2_start, length)
            merged_image = self.create_sequence_comparison_image(
                f"gifs/{vid1}:{vid1_start}_{vid2}:{vid2_start}_{length}s.gif",
                converter[vid1],
                converter[vid2],
                vid1_start,
                vid2_start,
                length,
            )
            merged_images.append(merged_image)
        return merged_images

    def convert_sequences_to_timestamps(self, vid1, vid2, verbose=False):
        """
        Convert the sequences to timestamps.

        The sequences are encoded according to diagonal number - see np.diag for details.

        In short, negative numbers are below the main diagonal, and positive numbers above.
        """
        sequences = self.found_sequences
        columns = [
            "Video 1",
            "Video 1 frame (S)",
            "Video 1 frame (HH:MM:SS)",
            "Video 2",
            "Video 2 frame (S)",
            "Video 2 frame (HH:MM:SS)",
            "Duration",
        ]

        timestamps_df = pd.DataFrame(columns=columns)

        timestamps = []

        conv_fun = self.neighbours._secs_to_timestamp

        for location, length in sequences[vid1][vid2].items():
            diagonal, offset = location
            if diagonal < 0:
                # Start below diagonal
                start_time = (diagonal * -1, 0)
            else:
                # Start above diagonal
                start_time = (0, diagonal)
            # Add offset in either case
            start_time = (start_time[0] + offset, start_time[1] + offset)

            timestamps.append((start_time, length))
            result = {
                k: v
                for k, v in zip(
                    columns,
                    [
                        vid1,
                        start_time[0],
                        conv_fun(start_time[0]),
                        vid2,
                        start_time[1],
                        conv_fun(start_time[1]),
                        length,
                    ],
                )
            }
            timestamps_df = timestamps_df.append(result, ignore_index=True)

        if verbose:
            for (vid1_start, vid2_start), length in timestamps:
                print(
                    f"{vid1}:{vid1_start}-{vid1_start+length}s -  {vid2}:{vid2_start}-{vid2_start+length}s"
                )

        return timestamps_df

    def _video_frame_to_global_index(self, video, frame):
        return min(self.neighbours.frames.cached_video_index[video]) + frame

    def _add_score_to_sequence_df(self, df, score_style):
        df = df.assign(Score="")
        new_sequence_list = []

        #     display(df)
        for sequence_dict in df.to_dict(orient="records"):
            vid1, vid2, vid1_s, vid2_s, duration = [
                sequence_dict[x]
                for x in [
                    "Video 1",
                    "Video 2",
                    "Video 1 frame (S)",
                    "Video 2 frame (S)",
                    "Duration",
                ]
            ]
            #         print(sequence_dict)
            #         return None
            dlist = self.neighbours.distance_list

            v1_global_frame = self._video_frame_to_global_index(vid1, vid1_s)
            v2_global_frame = self._video_frame_to_global_index(vid2, vid2_s)
            v1_frame_indexes = set(range(v1_global_frame, v1_global_frame + duration))
            v2_frame_indexes = set(range(v2_global_frame, v2_global_frame + duration))

            relevant_indexes = v1_frame_indexes.union(v2_frame_indexes)

            #         print(relevant_indexes)
            relevant_dlist = [(d, i) for d, i in dlist if i[0] in relevant_indexes]

            # Create expected frame pairs
            expected_pairs = list(
                [(v1_global_frame + i, v2_global_frame + i) for i in range(duration)]
            )

            # List of lists, first value being vid1->vid2, second being vid2->vid1 (TODO: Verify if they are always equal!!)
            distances = [-1 for i in range(duration)]

            for d, i in relevant_dlist:
                if i[0] in v1_frame_indexes:
                    distance_from_start = i[0] - v1_global_frame
                    matching_video_index = v2_global_frame + distance_from_start
                    relevant_match_distance = next(
                        iter(
                            [_d for _d, _i in zip(d, i) if _i == matching_video_index]
                        ),
                        None,
                    )
                    if relevant_match_distance is None:
                        distances[distance_from_start] = -1
                    else:
                        distances[distance_from_start] = relevant_match_distance
                else:
                    distance_from_start = i[0] - v2_global_frame
                    matching_video_index = v1_global_frame + distance_from_start
                    relevant_match_distance = next(
                        iter(
                            [_d for _d, _i in zip(d, i) if _i == matching_video_index]
                        ),
                        None,
                    )
                    if relevant_match_distance is None:
                        distances[distance_from_start] = -1
                    else:
                        distances[distance_from_start] = relevant_match_distance

            sequence_dict["Score"] = score_style(distances)  # CHANGE METHOD HERE
            new_sequence_list.append(sequence_dict)

        return pd.DataFrame(new_sequence_list).sort_values(by="Score")

    def get_all_sequences(self, score_function=np.median):
        videos = list(self.found_two_way_sequences.keys())
        other_videos = set(videos)
        all_video_sequences = []
        for vid in videos:
            other_videos -= {vid}
            #         print(other_videos)
            for ovid in list(other_videos):
                all_video_sequences.append(
                    self.convert_sequences_to_timestamps(vid, ovid, verbose=False)
                )
        df = pd.concat(all_video_sequences)
        return self._add_score_to_sequence_df(df, score_function)
