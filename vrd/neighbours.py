"""Handles logic relating to nearest neighbours, i.e. which frames are similar to
each other.
"""
import functools
import math
import os
import time
import warnings
from copy import deepcopy
from itertools import compress, repeat
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from .frame_extractor import FrameExtractor
from .image_preprocessing import is_monochrome


class Neighbours:
    """Neighbours class, containing values and logic for closest neighbours for frames
    in the VRD.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """

    frames: FrameExtractor
    distance_list: list

    def __init__(self, frames: FrameExtractor, distance_list):
        self.frames = frames
        self.distance_list = distance_list
        self._fix_neighbours()

    def copy(self):
        new_neighbours = Neighbours(self.frames.copy(), self.distance_list.copy())
        return new_neighbours

    def filter_same_video(self):
        """Filters any neighbors that belongs to the same video as the original frame.

        Raises:
            Exception: In case the image list does not match the distance list
        """
        all_images = self.frames.all_images

        # print(f'Number of frames: {len(all_images)}')
        # print(f'Length of distance list: {len(self.distance_list)}')
        if len(all_images) != len(self.distance_list):
            raise Exception(
                "Expected distance list length to match total number of images! Please run this filter first."
            )

        video_list = list(self.frames.cached_video_index.keys())
        #     print(video_list)

        total_cleaned = 0

        for vid in video_list:
            vid_indexes = list(self.frames.get_index_from_video_name(vid))
            if len(vid_indexes) == 0:
                print(
                    f"Unable to find Indexes matching video {vid}. Skipping for now..."
                )
                continue
            vid_index_min = vid_indexes[0]
            vid_index_max = vid_indexes[-1]

            for vid_index in vid_indexes:
                d, i = self.distance_list[vid_index]
                iarr = np.array(i)
                if len(iarr) == 0:
                    continue

                in_video = (iarr >= vid_index_min) & (iarr <= vid_index_max)
                in_video[0] = False  # Keep index 0!

                num_removed = np.sum(in_video)
                if num_removed > 0:
                    total_cleaned += 1

                # TODO: Make this quicker?
                d = [x for x, r in zip(d, in_video) if not r]
                i = [x for x, r in zip(i, in_video) if not r]
                self.distance_list[vid_index] = (d, i)

    def _fix_neighbours(self):
        """Ensures that if  index A is a neighbour to index B, the reverse is also true!"""
        distance_copy = deepcopy(self.distance_list)
        # lookup_table = {_i[0]:(i, set(_i[1:]), {__i:__d for __d, __i in zip(_d[1:],_i[1:])}) for i, (_d,_i) in enumerate(distance_copy)}
        lookup_table = {
            _i[0]: (i, set(_i[1:])) for i, (_d, _i) in enumerate(distance_copy)
        }
        added_count = 0
        for d, i in tqdm(distance_copy):
            # Get the original frame for this distance
            _, orig_i = d[0], i[0]
            # d_to_add = list()
            # i_to_add = list()
            for _d, _i in zip(d[1:], i[1:]):
                try:
                    neighbour_list_index, matching_set = lookup_table[_i]
                except KeyError:
                    # This happens if the neighbour frame has been deleted - due to lack of neighbours, or just filtered. Should we create it in this case?
                    # If it was filtered, this goes against the "wishes of the operator", so safest to skip for now.
                    # Best case, just run this just after creation; Doing so sidesteps this problem
                    continue
                # Check if the original frame is a neighbour in the neighbours' neighbour list
                if orig_i not in matching_set:
                    added_count += 1
                    # If not there, it should be added (including distance!)
                    # print(f'Len before: {len(self.neighbours.distance_list[neighbour_list_index][0])}')
                    new_d = np.append(
                        self.distance_list[neighbour_list_index][0], _d
                    )  # add original frame distance to neighbour as distance
                    new_i = np.append(
                        self.distance_list[neighbour_list_index][1], orig_i
                    )  # i
                    self.distance_list[neighbour_list_index] = (new_d, new_i)

        self.distance_list = sorted(self.distance_list, key=lambda x: x[1][0])
        # print(f"Added {added_count} to distance list.")

    def _get_subdir(self, index):
        frame = self.frames.all_images[index]
        return self.frames.get_subdir_for_frame(frame)

    def filter_neighbours_not_in_subfolder(self, folders, keep_reverse_matches=False):
        """Filters any neighbours that are not related to the specified subfolder.

        The keep_reverse_matches argument changes how this works:
        If set to False:
          Only frames in the specified subfolders are retained, all others are removed
        If set to True:
          The same as above, but additionally frames outside the subfolders,
          but with neighbours in one of the specified subfolders are retained

        Args:
            folders ([type]): A list of folders
            keep_reverse_matches (bool, optional): Whether or not to keep frames outside the subfolders but with neighbours in the subfolders.
        """
        dlist = self.distance_list
        if isinstance(folders, list):
            folders = set(folders)
        else:
            folders = set([folders])
        new_dlist = []
        for d, i in dlist:
            if keep_reverse_matches:
                if self._get_subdir(i[0]) in folders:
                    new_dlist.append((d, i))
                else:
                    new_d, new_i = [d[0]], [i[0]]
                    for dd, ii in zip(d[1:], i[1:]):
                        subdir = self._get_subdir(ii)
                        if subdir in folders:
                            new_d.append(dd)
                            new_i.append(ii)
                    if len(new_d) > 1:
                        new_dlist.append((new_d, new_i))
            elif self._get_subdir(i[0]) in folders:
                new_dlist.append((d, i))
        self.distance_list = new_dlist

    def filter_few_neighbours(self, min_neighbours: int):
        """Filter distance lists with < the specified amount of neighbours"""
        new_dlist = []
        count = 0
        for d, i in self.distance_list:
            if len(i) < (min_neighbours + 1):
                count += 1
                continue
            new_dlist.append((d, i))

        self.distance_list = new_dlist
        # print(f'Removed {count} frames with less than {min_neighbours} neighbours.')

    def get_best_match(self, video_1, video_2):
        """
        Gets the best video match between two videos, given the FrameExtractor, the associated distance_list,
        and the two video names.
        """
        v1_indexes = self.frames.get_index_from_video_name(video_1)
        v2_indexes = self.frames.get_index_from_video_name(video_2)
        v1_dlist = [x for x in self.distance_list if x[1][0] in v1_indexes]
        # print(Video_1)
        # print(Video_2)
        new_id_list = []

        for d, i in v1_dlist:
            is_v2_index = list(map(lambda x: x in v2_indexes, i))
            if np.sum(is_v2_index) == 0:  # Skip if there are no matches
                continue
            is_v2_index[0] = True  # Keep reference index!
            new_id_list.append(
                (list(compress(d, is_v2_index)), list(compress(i, is_v2_index)))
            )
        # print(f'Final list of matches: {len(new_id_list)}')
        # Now new_id_list is only matches with video 2! Sort by shortest distance!
        new_id_list = sorted(new_id_list, key=lambda x: x[0][1])
        if len(new_id_list) == 0:
            return None
        # first (and therefore best) index after sorting!
        relevant_index = new_id_list[0][1]

        return (
            self.frames.all_images[relevant_index[0]],
            self.frames.all_images[relevant_index[1]],
        )

    def _get_remaining_count(self):
        return np.array([(len(x) - 1) for x, y in self.distance_list])

    def plot_remaining_statistics(self, figure_size=(15, 6), **kwargs):
        """Displays histograms for 1) remaining neighbours and 2) remaining distances.

        Using these values, the user can detect if a filtering step removed too many or too few neighbours.
        """
        remaining_count = self._get_remaining_count()

        _, axes = plt.subplots(1, 2, figsize=figure_size)
        hist = sns.histplot(remaining_count, ax=axes[0], **kwargs)
        #     axes[0].set_xlim((0,max_value))

        hist.set(
            title="Histogram of number of remaining neighbours per frame",
            xlabel="Number of neighbours",
            ylabel="Number of frames",
        )

        dlist = self.distance_list
        dist_list = []
        for d, _ in dlist:
            dist_list.extend(d[1:])

        hist = sns.histplot(dist_list, ax=axes[1])
        hist.set(
            title="Histogram of neighbour distances",
            ylabel="Number of neighbours",
            xlabel="Neighbour distance",
        )
        plt.xticks(rotation=45)
        return hist

    def print_zero_remaining_neighbours(self):
        """Generate a dataframe containing per-video statistics of frames with zero neighbours

        Returns:
            pandas dataframe
        """
        zero_remaining_neighbour_videos = {}
        remaining_count = self._get_remaining_count()
        for i, count in enumerate(remaining_count):
            if count == 0:
                video_name = self.frames.get_video_name(self.frames.all_images[i])
                try:
                    zero_remaining_neighbour_videos[video_name] += 1
                except IndexError:
                    zero_remaining_neighbour_videos[video_name] = 1
        print(" Count: Video name:")

        df = pd.DataFrame(columns=["Frames with no self", "Video name"])

        for vid, count in zero_remaining_neighbour_videos.items():
            df.append(
                {"Frames with no self": count, "Video name": vid}, ignore_index=True
            )

        return df

    def get_video_match_statistics(self, distance_threshold=20000):
        """Creates and returns a dataframe containing the statistics for each video

        The output includes total valid matches, total amount of frames and video name for each video.

        Args:
            distance_threshold (int, optional): The maximum allowed threshold for distance. Defaults to 20000.
        """
        video_list = list(self.frames.cached_video_index.keys())

        if distance_threshold < 0:
            # TODO: Verify if this int max is enough
            distance_threshold = 9223372036854775807

        dlist_sorted = sorted(self.distance_list, key=lambda x: x[1][0])  # Sort index
        dlist_indexes = np.array([x[1][0] for x in dlist_sorted])

        dict_list = []
        columns = [
            "Video name",
            "Original no. of frames",
            "Remaining no. of frames",
            "Remaining no. of neighbours",
            "Average no. of neighbours",
        ]

        for vid in video_list:
            vid_indexes = self.frames.get_index_from_video_name(vid)
            if vid_indexes is None or len(vid_indexes) == 0:
                print(f"No indexes for video {vid}, continuing...")
                continue
            vid_index_min = min(vid_indexes)
            vid_index_max = max(vid_indexes)

            try:
                vid_loc = np.where(
                    (dlist_indexes >= vid_index_min) & (dlist_indexes <= vid_index_max)
                )[0]
            except:
                print(f"Error finding location of video {vid}, continuing...")
                continue
            if len(vid_loc) == 0:
                # Video has likely been completely filtered from the distance list.
                dict_list.append(
                    {x: y for x, y in zip(columns, [vid, len(vid_indexes), 0, 0, 0])}
                )
                continue

            vid_dlist = dlist_sorted[vid_loc[0] : vid_loc[-1]]

            # This checks for number of matches where distance < distance_threshold
            vid_distances = [
                (np.sum(np.array(x) < distance_threshold) - 1) for x, y in vid_dlist
            ]

            if len(vid_distances) == 0:
                mean_dist = 0
            else:
                mean_dist = np.mean(vid_distances)

            dict_list.append(
                {
                    x: y
                    for x, y in zip(
                        columns,
                        [
                            vid,
                            len(vid_indexes),
                            len(vid_distances),
                            np.sum(vid_distances),
                            mean_dist,
                        ],
                    )
                }
            )
        return pd.DataFrame(dict_list, columns=columns)

    def filter_monochrome_images(self, pool_size=6, allowed_difference=40):
        """Filters any images that have a "difference" lower than the specified threshold.

        This is intended to remove frames that are of a solid color.

        For details see get_monochrome_images.

        Args:
            pool_size (int, optional): [description]. Defaults to 6.
            allowed_difference (int, optional): [description]. Defaults to 40.
        """
        image_indexes = self.get_monochrome_frames(
            pool_size=pool_size, allowed_difference=allowed_difference
        )
        self.remove_indexes_from_distance_list(image_indexes)
        return self
        # print(f'Removed {len(image_indexes)} images from the distance list.')

    def get_monochrome_frames(self, pool_size=6, allowed_difference=40):
        """Get the indexes (as determined by all_images) of all images with largely the same color.

        This function is threaded to avoid bottlenecks.

        For details, see image_is_same_color.

        Args:
            distance_list ([type]): The distance list used to look for relevant images
            all_images ([type]): An ordered list of all possible images
            pool_size (int, optional): How many threads to used. Defaults to 6.
            allowed_difference (int, optional): Allowed difference in the images. Defaults to 40.

        Returns:
            set: A set containing the indexes (corresponding to the location in all_images) of images deemed to have a single color
        """
        pool = Pool(pool_size)

        # remaining indexes after filtering
        relevant_indexes = [x[1][0] for x in self.distance_list]
        relevant_files = [self.frames.all_images[x] for x in relevant_indexes]

        try:
            pool_result = pool.starmap(
                is_monochrome, zip(relevant_files, repeat(allowed_difference))
            )
        finally:
            pool.close()
            pool.join()
        # Create array to enable indexing; create set as it is much faster when looking up later
        monochrome_frames = set(np.array(relevant_indexes)[np.nonzero(pool_result)[0]])
        return monochrome_frames

    def remove_indexes_from_distance_list(self, image_indexes: set):
        """Remove the indexes defined in the image_indexes set from the distance_list.

        This means that they are removed both if they are the oringinal (reference) image,
        and also when they are the neighbour.

        Image indexes is a set as it's faster.

        Args:
            distance_list (list): The faiss distance list to remove from
            image_indexes (set): A set of indexes to remove

        Returns:
            [list]: A distance list with the specified image indexes removed.
        """
        dlist = []
        for d, i in self.distance_list:
            if i[0] in image_indexes:
                continue
            new_i = []
            new_d = []
            for distance, index in zip(d, i):
                if index in image_indexes:
                    continue
                new_d.append(distance)
                new_i.append(index)
            if len(new_d) > 1:
                dlist.append((new_d, new_i))

        self.distance_list = dlist
        return self

    def filter_neighbours_in_same_subfolder(self, pool_size=-1, batch_size=-1):
        """Filter neighbours that are from the same subdirectory as the source frame."""
        new_distance_list = []

        if pool_size < 1:
            pool_size = os.cpu_count()
            if pool_size is None:
                pool_size = 1

        pool = Pool(pool_size)

        if batch_size <= 0:
            # Default value, just split it equally in one batch per thread.
            batch_size = math.floor(len(self.distance_list) / pool_size + 1)

        # Step 1: Calculate all subdirs
        new_iter = (
            [x, self.frames.frame_directory]
            for x in self._batch(self.frames.all_images, batch_size)
        )
        subdirs = {}
        for img_subdir in pool.imap_unordered(self._get_subdir_for_frame, new_iter):
            for img, subdir in img_subdir:
                subdirs[img] = subdir

        # print(f'Got {len(subdirs.keys())} subdirs...')

        new_iter = (
            [x, self.frames.all_images, subdirs]
            for x in self._batch(self.distance_list, batch_size)
        )
        new_distance_list = []

        # NOTE: This was actually slower using multiprocessing pools, likely because too much was being serialized.
        # Reconsider if required, but this approach now is fairly efficient due to precalculations.
        for work in new_iter:
            partial_dlist = self._same_subdir_fun(work)
            new_distance_list.extend(partial_dlist)

        self.distance_list = new_distance_list
        return self

    def filter_maximum_distance(self, max_distance):
        """Filters all neighbours above the specified maximum distance

        Args:
            max_distance ([int]): The maximum distance as a number

        Returns:
            self
        """
        # TODO: Use slices instead, saves memory?
        new_dlist = []
        for d, i in self.distance_list:
            new_d = []
            new_i = []
            for _d, _i in zip(d, i):
                if _d > max_distance:
                    break
                new_d.append(_d)
                new_i.append(_i)
            if len(new_d) > 1:
                new_dlist.append((new_d, new_i))
        self.distance_list = new_dlist
        return self

    @staticmethod
    def _get_subdir_for_frame(args):
        images, frame_directory = args

        # TODO: THIS IS NOT CORRECT; FIX! THIS REFERENCES A CLASS METHOD
        return [
            (img, FrameExtractor.get_subdir_for_frame_static(img, frame_directory))
            for img in images
        ]

    # as per https://stackoverflow.com/a/8290508
    @staticmethod
    def _batch(iterable, batch_size=1):
        iter_length = len(iterable)
        for ndx in range(0, iter_length, batch_size):
            yield iterable[ndx : min(ndx + batch_size, iter_length)]

    @staticmethod
    def _same_subdir_fun(args):
        """Function for calculating if neighoburs are from the same subdirectory.

        This is used by the pool function to allow multiprocessing and should not be directly called!
        """
        d_is, all_images, subdirs = args
        new_dlist = []
        for d, i in d_is:
            ref_subdirectory = subdirs[all_images[i[0]]]

            if ref_subdirectory is None:
                continue

            # always add reference first (as it will be removed at it has the same subdir as itself)
            cleaned_di = list([(d[0], i[0])])

            for _d, _i in zip(d, i):
                if subdirs[all_images[_i]] != ref_subdirectory:
                    cleaned_di.append((_d, _i))
            if len(cleaned_di) < 2:
                continue
            new_di = ([x[0] for x in cleaned_di], [x[1] for x in cleaned_di])
            new_dlist.append(new_di)
        # Convert to correct form of one tuple of two lists

        return new_dlist

    def filter_neighbours(self, filter_list: list, create_copy=True):
        """Applies the specified functions to filter the list of self.

        Args:
            filter_list (list): A list of the filters, essentially separate functions
            create_copy (bool, optional): Whether or not to create a copy of the neighbour list.

        Returns:
            The filtered list
        """
        if filter_list is None:
            return self

        if create_copy:
            return_copy = self.copy()
        else:
            return_copy = self

        for filter in filter_list:
            return_copy = filter(return_copy)

        return return_copy

    def _find_distance_between_two_frames(self, frame_1, frame_2):
        dlist = self.distance_list

        f1_d, f1_i = next(
            iter([(d, i) for d, i in dlist if i[0] == frame_1]), (None, None)
        )
        f2_d, f2_i = next(
            iter([(d, i) for d, i in dlist if i[0] == frame_2]), (None, None)
        )

        # Consider if making the distance negative makes sense if it isn't found...
        if f1_d is None or f2_d is None:
            return (-1, -1)

        f1_dist = -1
        f2_dist = -1

        for d, i in zip(f1_d, f1_i):
            if i == frame_2:
                f1_dist = d
                break

        for d, i in zip(f2_d, f2_i):
            if i == frame_1:
                f2_dist = d
                break

        return (f1_dist, f2_dist)

    def get_top_frames(self, number_to_return=20) -> pd.DataFrame:
        """Get the top frames ith most remaining matching self.
        The information includes: Video name, timestamp, and number of self.

        Args:
            number_to_return (int, optional): [description]. Defaults to 20.

        Returns:
            DataFrame: A dataframe with the results
        """
        dlist = self.distance_list
        frames = self.frames

        index_and_count = [(idx, len(i) - 1) for idx, (d, i) in enumerate(dlist)]
        sort = sorted(index_and_count, key=lambda idx_count: idx_count[1], reverse=True)
        to_return = sort[0:number_to_return]
        return_list = []
        for idx, count in to_return:
            vid, start_time = frames.get_video_and_start_time_from_index(idx)

            return_list.append(
                (vid, start_time, self._secs_to_timestamp(start_time), count)
            )
        df = pd.DataFrame(
            return_list,
            columns=[
                "Video name",
                "Frame (S)",
                "Frame (HH:MM:SS)",
                "Remaining neighbours",
            ],
        )
        df.index += 1
        # df = df.style.set_properties(**{'text-align': 'left'})
        return df

    @staticmethod
    def _secs_to_timestamp(secs):
        return time.strftime("%H:%M:%S", time.gmtime(secs))

    def get_frames_with_closest_distance(self, number_to_return=10):
        """Get frames with closest distances to other frames.

        This automatically removed suplicates, i.e if frame1:frame2 and frame2:frame1.

        Args:
            number_to_return (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """
        frames = self.frames
        dlist = self.distance_list
        return_list = []
        for d, i in dlist:
            ref_index = i[0]
            return_list.extend([(ref_index, dd, ii) for dd, ii in zip(d[1:], i[1:])])
            return_list = sorted(return_list, key=lambda x: x[1], reverse=False)

            # Add check to ensure vid1:vid2 and vid2:vid1 are not present at the same time
            indexes_to_remove = []
            last_distance = -1
            for list_index, (vid1_idx, dist, vid2_idx) in enumerate(return_list):
                if dist == last_distance:
                    last_vid1_idx, _, last_vid2_idx = return_list[list_index - 1]
                    if last_vid2_idx == vid1_idx and last_vid1_idx == vid2_idx:
                        indexes_to_remove.append(list_index)
                last_distance = dist

            return_list = [
                x for i, x in enumerate(return_list) if i not in indexes_to_remove
            ]

            if len(return_list) > number_to_return:
                return_list = return_list[0:number_to_return]
        columns = [
            "Video 1 name",
            "Video 1 frame (S)",
            "Video 1 frame (HH:MM:SS)",
            "Video 2 name",
            "Video 2 frame (S)",
            "Video 1 frame (HH:MM:SS)",
            "Distance",
        ]
        df_list = []

        for v1_index, distance, v2_index in return_list:
            vid1, frame_start1 = frames.get_video_and_start_time_from_index(v1_index)
            vid2, frame_start2 = frames.get_video_and_start_time_from_index(v2_index)
            ts1 = self._secs_to_timestamp(frame_start1)
            ts2 = self._secs_to_timestamp(frame_start2)

            df_list.append(
                {
                    k: v
                    for k, v in zip(
                        columns,
                        [vid1, frame_start1, ts1, vid2, frame_start2, ts2, distance],
                    )
                }
            )

        df = pd.DataFrame(df_list, columns=columns)
        df.index += 1
        # df_styler = df.style.set_properties(**{'text-align': 'left'})
        # df_styler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

        return df


# class BuiltInFilters:
#     def __only_from_video(self: Neighbours, video_name):
#         distances, indexes = self.distance_list
#         allowed_indexes = set(self.frames.get_index_from_video_name(video_name))
#         new_distance_list = []

#         # I consider this more readable than the list comprehension variant...
#         for _d, _i in zip(distances, indexes):
#             if _i[0] in allowed_indexes:
#                 new_distance_list.append((_d, _i))

#         self.distance_list = new_distance_list
#         return self

#     def only_from_video(video_name):
#         return lambda self: BuiltInFilters.__only_from_video(self, video_name)


class FilterChange:
    """Show changes in statistics compared to the initial state.

    Returns:
        [type]: [description]
    """

    orig_frame_count: int
    curr_frame_count: int
    reference_distance_list: list

    def __init__(self, neighbours: Neighbours) -> None:
        self.orig_frame_count = len(neighbours.frames.all_images)
        self.curr_frame_count = len(neighbours.distance_list)
        self.reference_distance_list = deepcopy(neighbours.distance_list)

    def per_video_comparison(self, neighbours_new: Neighbours) -> pd.DataFrame:
        """Compares the initial state to the new state.

        Used to shown what impact a filtering step had.

        This function provide this information per step:

        Video name, total frames, lost frames, remaining frames, Average number of self before, Average number of self after

        Args:
            neighbours_new ([Neighbours]): The new self state, generarally after filtering

        Returns: A pandas dataframe containing the statistics.

        """
        df = pd.DataFrame(
            columns=[
                "Video name",
                "Total frames",
                "Lost frames",
                "Remaining frames",
                "Avg. self before",
                "Avg. self after",
            ]
        )
        for video, video_indexes in neighbours_new.frames.cached_video_index.items():
            ref_dlist = [
                (d, i) for d, i in self.reference_distance_list if i[0] in video_indexes
            ]
            new_dlist = [
                (d, i) for d, i in neighbours_new.distance_list if i[0] in video_indexes
            ]
            avg_neighbours_ref = 0
            avg_neighbours_new = 0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_neighbours_ref = np.mean([len(i) - 1 for d, i in ref_dlist])
                avg_neighbours_new = np.mean([len(i) - 1 for d, i in new_dlist])

            total_frames = len(video_indexes)
            lost_frames = len(ref_dlist) - len(new_dlist)

            df = df.append(
                {
                    "Video name": video,
                    "Total frames": total_frames,
                    "Lost frames": lost_frames,
                    "Remaining frames": len(new_dlist),
                    "Avg. self before": avg_neighbours_ref,
                    "Avg. self after": avg_neighbours_new,
                },
                ignore_index=True,
            )
        return df

    def changes_in_short(self, neighbours) -> pd.DataFrame:
        """
        Total frames:
        Remaining Frames (before)
        Remaining Frames (after)
        Lost frames
        Average neighbours (before)
        Average neighbours (after)
        Removed percentage

        Args:
            neighbours ([type]): [description]

        Returns:
            pd.DataFrame: [description]
        """
        df = pd.DataFrame(columns=["Before this filter", "After this filter"])

        def create_row(vals, name):
            return pd.Series({k: v for k, v in zip(df.columns, vals)}, name=name)

        remaining_before = self.curr_frame_count
        remaining_after = len(neighbours.distance_list)
        df = df.append(
            create_row([remaining_before, remaining_after], "Remaining frames")
        )

        lost_frames = remaining_before - remaining_after
        df = df.append(create_row([np.NaN, lost_frames], "Lost frames (number)"))

        removed_percentage = (lost_frames / remaining_before) * 100
        df = df.append(
            create_row([np.NaN, removed_percentage], "Lost frames (percentage)")
        )

        avg_neighbours_before = np.sum(
            [len(i) - 1 for d, i in self.reference_distance_list]
        ) / len(self.reference_distance_list)
        avg_neighbours_after = np.sum(
            [len(i) - 1 for d, i in neighbours.distance_list]
        ) / len(neighbours.distance_list)
        df = df.append(
            create_row(
                [avg_neighbours_before, avg_neighbours_after],
                "Average no. of neighbours",
            )
        )

        # Calculate average distances
        def get_average_distance(dlist):
            distances = [(len(d) - 1, np.sum(d[1:])) for d, i in dlist]
            if len(distances) == 0:
                return -1

            tot_count, tot_distance = functools.reduce(
                lambda d1, d2: (d1[0] + d2[0], d1[1] + d2[1]), distances
            )

            return tot_distance / tot_count

        avg_distance_before = get_average_distance(self.reference_distance_list)
        avg_distance_after = get_average_distance(neighbours.distance_list)
        df = df.append(
            create_row(
                [
                    avg_distance_before,
                    avg_distance_after if avg_distance_after > -1 else "-",
                ],
                "Average distance metrics",
            )
        )
        df[df.columns[0]] = df[df.columns[0]].apply(lambda v: np.round(v, 2))
        df = df.fillna("-")
        return df

    def overall_comparison(self, neighbours_new: Neighbours) -> pd.DataFrame:
        """Compares two neighbours to see how much has been changed."""
        df = pd.DataFrame(columns=["Total frames", "Lost frames", "Remaining frames"])

        ref_dlist = deepcopy(self.reference_distance_list)
        new_dlist = deepcopy(neighbours_new.distance_list)
        avg_neighbours_ref = 0
        avg_neighbours_new = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_neighbours_ref = np.mean([len(i) - 1 for d, i in ref_dlist])
            avg_neighbours_new = np.mean([len(i) - 1 for d, i in new_dlist])

        total_frames = len(neighbours_new.frames.all_images)
        lost_frames = len(ref_dlist) - len(new_dlist)

        df = df.append(
            {
                "Total frames": total_frames,
                "Lost frames": lost_frames,
                "Remaining frames": len(new_dlist),
                "Avg. self before": avg_neighbours_ref,
                "Avg. self after": avg_neighbours_new,
            },
            ignore_index=True,
        )

        return df
