# Load annotation
import ntpath
import re
from itertools import repeat
from multiprocessing import Pool

import pandas as pd


class AnnotationMatcher:
    """A class allowing for reading the annotation files belonging to the video reuse dataset

    Returns:
        AnnotationMatcher: An annotation matcher
    """

    df: pd.DataFrame
    cached_df: pd.DataFrame
    cached_video: str
    video_df_cache: dict
    pairwire_matches: dict

    @property
    def video_list(self):
        """Gets a sorted video list of the included video files in the annotation set

        Returns:
            list: A list containing the names (as strings) of the included video files
        """
        return sorted(list(self.video_df_cache.keys()))

    def __init__(self, annotation_directory, annotation_name):
        """Initialize the AnnotationMatcher

        Args:
            annotation_directory (string): A path to the base directory of the annotation files
            annotation_name ([type]): The name of the specific annotation dataset
        """
        self.df = self.load_annotation(annotation_directory, annotation_name)
        self.cached_df = None
        self.cached_video = None
        self.video_df_cache = {}
        self.pairwire_matches = {}
        self._cache_all_matches()

    def load_annotation(self, annotation_directory, annotation_name):
        """Loads the annotations

        Args:
            annotation_directory ([type]): [description]
            annotation_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        df = pd.read_csv(annotation_directory + annotation_name, header=None)
        df.columns = [
            "Video_1",
            "Video_2",
            "Start_video_1",
            "End_video_1",
            "Start_video_2",
            "End_video_2",
        ]
        print("Start replace...")
        df[["Start_video_1", "End_video_1", "Start_video_2", "End_video_2"]] = df[
            ["Start_video_1", "End_video_1", "Start_video_2", "End_video_2"]
        ].applymap(self._timestamp_to_seconds)
        print("End replace...")
        #     df['Video 1'] = df['Video 1'].replace('\.[a-zA-Z0-9]+','',regex=True)
        #     df['Video 2'] = df['Video 2'].replace('\.[a-zA-Z0-9]+','',regex=True)
        return df

    def get_pairwise_matches(self, vid1, vid2):
        try:
            return self.pairwire_matches[(vid1, vid2)]
        except KeyError:
            vid1_df = self.get_relevant_matches(vid1)
            self.pairwire_matches[(vid1, vid2)] = vid1_df.loc[vid1_df.Video_2 == vid2]

    def _cache_all_matches(self):
        df = self.df
        unique_videos = (df.Video_1.append(df.Video_2)).unique()

        for vid in unique_videos:
            vid1_df = self.get_relevant_matches(vid)
            video_2s = vid1_df.Video_2.unique()
            for vid2 in video_2s:
                self.get_pairwise_matches(vid, vid2)

    # def has_overlap(df, video_1, video_2)

    def get_relevant_matches(self, video):
        df = self.df
        if video in self.video_df_cache:
            return self.video_df_cache[video]
        #         print(f'Video: {video}')
        #         print(f'Cached: {self.cached_video}')
        # Simply ignore self-references for now!
        df = df[df.Video_1 != df.Video_2]

        df1 = df[df.Video_1 == video]
        v2_loc = df.Video_2 == video
        if not any(v2_loc):
            self.video_df_cache[video] = df1
            return df1
        df2 = df.loc[v2_loc]

        df2[["Video_1", "Video_2"]] = df2[["Video_2", "Video_1"]]
        df2[["Start_video_1", "Start_video_2"]] = df2[
            ["Start_video_2", "Start_video_1"]
        ]
        df2[["End_video_1", "End_video_2"]] = df2[["End_video_2", "End_video_1"]]
        temp = df1.append(df2)
        temp = temp[~temp.index.duplicated(keep="first")]

        self.video_df_cache[video] = temp
        return temp

    @staticmethod
    def _is_within(s: int, e: int, sec: int) -> bool:
        return (sec >= s) & (sec <= e)

    @staticmethod
    def _get_second_from_filename(filename: str) -> int:
        # Note: Requires file name to stay the same!
        return int(re.findall(r"frame_([0-9]+).png$", filename)[0])

    def are_overlapping(
        self, ref_frame: str, other_frame: str, frames_per_second=1
    ) -> bool:
        """
        Checks if the specified frames are overlapping, if so return True

        This method assumes that the frames can be converted directly to time,
        with a factor described by frames_per_second.
        """
        ref_video = ntpath.basename(ref_frame).split("_frame_")[0]
        other_video = ntpath.basename(other_frame).split("_frame")[0]

        reference_second = self._get_second_from_filename(ref_frame) / frames_per_second
        other_second = self._get_second_from_filename(other_frame) / frames_per_second

        df = self.get_pairwise_matches(ref_video, other_video)

        if df is None or len(df.index) == 0:
            return False

        for index, match in df.iterrows():

            r_s = match["Start_video_1"]
            r_e = match["End_video_1"]
            o_s = match["Start_video_2"]
            o_e = match["End_video_2"]
            if self._is_within(r_s, r_e, reference_second) & self._is_within(
                o_s, o_e, other_second
            ):
                return True
        return False

    @staticmethod
    def _timestamp_to_seconds(timestamp: str):
        try:
            hh, mm, ss = timestamp.split(":")
        except Exception as e:
            print("...")
            print(timestamp)
            print(e)
            return -1
        return int(hh) * 3600 + int(mm) * 60 + int(ss)


def __check_overlap_loop(d_i, all_images, am, invert):
    """
    Helper function for multiprocessing
    """
    D, I = d_i
    # Index 0 is always the reference video, so we keep that one.
    new_D = [D[0]]
    new_I = [I[0]]

    # Then loop index 1+ to look for videos that are not the same.
    for d, i in zip(D[1:], I[1:]):
        if am.are_overlapping(all_images[I[0]], all_images[i]) == invert:
            continue
        new_I.append(i)
        new_D.append(d)
    if len(new_D) < 2:  # Only if we ever found others...
        return (None, None)
    return (new_D, new_I)


def clean_distance_list_mp(
    am, distance_list: dict, all_images: list, pool_size=6, sort=False, invert=False
):
    """
    Only retains results from the distance list that are considered overlapping
    according to the annotation.

    If invert=True, instead only return results where there is no match.
    """
    pool = Pool(pool_size)
    pool_result = None

    try:
        pool_result = pool.starmap(
            __check_overlap_loop,
            zip(distance_list, repeat(all_images), repeat(am), repeat(invert)),
        )
    finally:
        pool.close()
        pool.join()

    res = [x for x in pool_result if x[0] is not None]
    if sort:
        # sort with shortest distance first
        return sorted(res, key=lambda x: x[0][1])
    return res
