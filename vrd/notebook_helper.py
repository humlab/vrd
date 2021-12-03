"""A mixed set of helper functions for the Jupyter notebooks, e.g. widgets
and image handling.
"""
import os
import random
import time
from inspect import Traceback
from itertools import compress
from multiprocessing import Pool
from typing import TYPE_CHECKING

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import PIL.ImageFont
import seaborn as sns
from ipyaggrid import Grid
from IPython.display import display
from matplotlib.backend_bases import MouseButton
from matplotlib.font_manager import findSystemFonts
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm

from .annotation_matcher import AnnotationMatcher
from .frame_extractor import FrameExtractor
from .image_preprocessing import process_image
from .neighbours import FilterChange
from .neural_networks import Network

if TYPE_CHECKING:
    from . import dbhandler

"""This file contains multiple helpful functions for the jupyter notebook, including showing best matches and so on.
"""


def get_video_name(video_string: str):
    """Get the short name of a video from a full video path.

    TODO: Make this work in windows as well! Consider using the version in thumbnail_generator only!

    Returns:
        string: The video name, without extension
    """

    return os.path.dirname(video_string).split("/")[-1]


def show_similar_videos(
    distance, indexes, image_paths, distance_threshold=200000, display_names=5
):
    """Show similar videos for a specific image file.

    Only the best match (per video) will be shown.
    No matches from the same video as the requested image will be shown.

    The results can be limited by number to show, as well as distance from the requested image.

    Args:
        distance (list): The list of distances - matches with indexes.
            The first one is the reference frame, expected to be distance 0.
        indexes (list): The list of indexes - matches with distances.
            The first one is the reference frame.
        image_paths (list): The list of image paths, where the index matches the indexes list
        distance_threshold (int, optional): A distance filter; do not show results with distances
            greater than this value. Defaults to 200000.
        display_names (int, optional): How many matches to display. Defaults to 5.
    """
    img = image_paths[indexes[0]]  # The reference frame
    print(f"reference frame: {img}")
    reference_video_name = get_video_name(img)
    names_to_show = []
    files_used = {}
    # Note: skip the first one as it's the reference frame
    for current_index, other_index in enumerate(indexes[1:], 1):
        name = image_paths[other_index]
        video_name = get_video_name(name)
        if video_name == reference_video_name:
            continue
        if video_name in files_used:  # Only show one per video
            continue
        files_used[video_name] = True

        names_to_show.append((name, distance[current_index]))
    names_to_show = names_to_show[:display_names]
    if len(names_to_show) == 0:
        return
    if names_to_show[0][1] > distance_threshold:
        return

    print("------------------")
    print(f"{img:40}:")
    for name, cnn_distance in names_to_show:
        print(f"({cnn_distance:.2f}, {name:30}")

    fig = plt.figure()
    ax = fig.add_subplot(161)
    ax.imshow(mpimg.imread(img))

    for j, subplot in enumerate(range(162, 162 + len(names_to_show))):
        name, cnn_distance = names_to_show[j]
        if cnn_distance > distance_threshold:
            break
        ax = fig.add_subplot(subplot)
        ax.imshow(mpimg.imread(name))
    plt.show()
    print("------------------")


def calculate_overlapping_video_statistics(
    am: AnnotationMatcher, frames: FrameExtractor, distance_list: list
):
    """
    Statistics include:
     - Total number of overlaps between video segments
     - Frames which had at least (any) match
     - Percentage of frames which had any match

    Note that having even a single match between video segments can be considered success,
    i.e. any_match > 0

    TODO: Consider applying multiprocessing to this, as it can take a long time.
    """
    video_df = None
    # Loop all available videos...
    for video in tqdm(sorted(am.video_df_cache.keys())):
        video_df = am.video_df_cache[video]
        if video_df is None:
            raise Exception(f"No cache find for video {video}!")
        try:
            video_df.loc[:, "total_matches"] = 0
            video_df.loc[:, "any_match"] = 0
            video_df.loc[:, "perc_frames_matched"] = 0
        except Exception:
            print(f"Broken video: {video}")

        other_videos = sorted(video_df.Video_2.unique())  # Get all other videos
        video_indexes = frames.get_index_from_video_name(video)
        # These are the distances for the current video
        video_distance_list = [x for x in distance_list if x[1][0] in video_indexes]
        video_distance_list = sorted(video_distance_list, key=lambda x: x[1][0])

        for o_vid in other_videos:  # Loop over other videos and find matches
            # print('Other video:')
            # print(o_vid)
            o_vid_df = video_df[video_df.Video_2 == o_vid]
            o_vid_indexes = frames.get_index_from_video_name(o_vid)
            # Find the index matching the current video in all_images
            if len(o_vid_indexes) == 0:
                print(f"All images length: {len(frames.all_images)}")
                raise Exception(f"No indexes found for file {o_vid}")
            for index, row in o_vid_df.iterrows():
                min_video = min(video_indexes)
                min_o_vid = min(o_vid_indexes)
                # Check if the frames matching video_indexes[second:end_second] have matches in dlist2 which match o_vid_indexes[Start_video_2:End_video_2]
                video_start_index = row.Start_video_1 + min_video
                video_end_index = row.End_video_1 + min_video
                o_vid_start_index = row.Start_video_2 + min_o_vid
                o_vid_end_index = row.End_video_2 + min_o_vid
                # Count the number of matches, as detected in the distance list!
                # NOTE: This is from the direction of Video 1, e.g. video 1 -> video 2. Video 2 has a separrate post later
                # TODO: Consider checking if video 1 results match with video 2 results, i.e. if Video 1 found a frame in video 2, will video 2 find the same frame in video 1?
                total_match_count = 0
                any_match_count = 0
                # We find the first index of the current range TODO: CACHE THIS????
                video_range = set(range(video_start_index, video_end_index))
                # Further refine the list and only keep the current range -  TODO: Cache?
                video_matches = [
                    x for x in video_distance_list if x[1][0] in video_range
                ]
                if len(video_matches) == 0:
                    continue

                for (
                    d_i
                ) in video_matches:  # Only loop over the relevant frames in video 1
                    # d_i = next((x for x in distance_list if x[1][0] == vid1_frame), None) # Find the distance matching vid1_frame in the distance list
                    if d_i is not None:
                        all_matched_indexes = np.array(sorted(d_i[1][1:]))
                        found_matches = np.sum(
                            (all_matched_indexes >= o_vid_start_index)
                            & (all_matched_indexes <= o_vid_end_index)
                        )
                        total_match_count += found_matches
                        any_match_count += found_matches > 0
                video_df.loc[index, "total_matches"] = total_match_count
                video_df.loc[index, "any_match"] = any_match_count
                video_df.loc[index, "perc_frames_matched"] = any_match_count / (
                    video_end_index - video_start_index
                )
    return video_df


def create_overlap_matrix(annot_match: AnnotationMatcher):
    """Create the overlap matrix, an all-vs-all matrix of how many
    of the expected annotations were correctly matched.

    Note that the value of video1 vs video2 may differ from the value of video2 vs video1, i.e. it is not symmetric.
    However, either way is likely to produce simlar results except in special cases.

    The source video is described on the Y axis, and the target on the X axis;
    for example, matrix[0][1] would be how many matches were correctly matched in video1 when video0 was the source.

    The value is ranges from 0-1, where 1 is all were matched and 0 is none were matched.

    Args:
        am (AnnotationMatcher): The annotation matcher

    Returns:
        np.array: The overlap matrix
    """
    video_list = annot_match.video_list
    overlap_matrix = np.empty((len(video_list), len(video_list)))
    overlap_matrix[:] = np.NaN
    for v1_index, v1 in enumerate(video_list):
        v1_df = annot_match.video_df_cache[v1]
        # v1_index = video_list.index(v1)
        for v2 in v1_df["Video_2"].unique():
            v2_df = v1_df[v1_df.Video_2 == v2]
            if len(v2_df.index) == 0:
                continue

            v2_index = video_list.index(v2)
            # TODO: MEAN is not really correct here; the different rows can have different number of values.
            overlap_matrix[v1_index, v2_index] = v2_df["perc_frames_matched"].mean()
    return overlap_matrix


def show_overlap_matrix(
    video_list,
    overlap_matrix,
    figsize=(14, 11),
    frames=None,
    am=None,
    distance_list=None,
    click_listener=False,
):
    """Shows an overlap matrix as a square heatmap.

    Args:
        video_list ([type]): [description]
        overlap_matrix ([type]): [description]
        figsize (tuple, optional): [description]. Defaults to (14, 11).
        thumbnails ([type], optional): [description]. Defaults to None.
        am ([type], optional): [description]. Defaults to None.
        distance_list ([type], optional): [description]. Defaults to None.
        click_listener (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    figure, ax = plt.subplots(figsize=figsize)
    heatmap_arr = overlap_matrix.copy()
    is_zero = heatmap_arr > 0  # COMPARE WITH NAN
    sns.heatmap(
        heatmap_arr,
        ax=ax,
        cmap=sns.color_palette("viridis", as_cmap=True),
        vmax=1,
        center=0.5,
        xticklabels=video_list,
        yticklabels=video_list,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        picker=0,
    )
    sns.heatmap(
        heatmap_arr,
        ax=ax,
        mask=is_zero,
        cmap="tab20c_r",
        cbar=False,
        vmax=1,
        center=0.5,
        xticklabels=video_list,
        yticklabels=video_list,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        picker=0,
    )  # The gray boxes

    if click_listener:
        # These need to be specified for this to work...
        assert frames is not None
        assert am is not None
        assert distance_list is not None

        def __overlap_onpick(event):
            if event.mouseevent.button is not MouseButton.LEFT:
                return
            x = -1
            y = -1
            vid_list = None
            try:
                vid_list = am.video_list
                x = int(event.ind % event.artist._meshWidth)
                y = int(event.ind // event.artist._meshWidth)
                txt = f"Event: {event}\nArtist: {event.artist}\n{event.ind}\nX: {x}\nY: {y}"
            except Exception:
                Traceback.print_exc()
            text.set_text(txt)
            try:
                text.set_text(
                    f"Video 1: {str(vid_list[y])}\nVideo 2:{str(vid_list[x])}"
                )
            except Exception:
                Traceback.print_exc()
            try:
                res = get_best_match(am, vid_list[y], vid_list[x])
                if res is not None:
                    vid1_file, vid2_file = res
                    print(f"Vid1_frame: {vid1_file}\nVid2_frame: {vid2_file}")
            except Exception:
                Traceback.print_exc()
            try:
                ab.set_visible(True)
                im1 = cv2.imread(vid1_file)
                im2 = cv2.imread(vid2_file)
                dim = (100, 100)
                im1_resized = cv2.resize(im1, dim)
                im2_resized = cv2.resize(im2, dim)
                img_to_show = cv2.hconcat([im1_resized, im2_resized])
                image.set_data(img_to_show)
            except Exception:
                Traceback.print_exc()

        text = ax.text(0, 0, "", va="bottom", ha="left")
        image = OffsetImage(mpimg.imread(frames.all_images[0]), zoom=1)
        xybox = (50.0, 50.0)
        ab = AnnotationBbox(
            image,
            (26, 0.1),
            xybox=xybox,
            xycoords="data",
            boxcoords="offset points",
            pad=0.1,
            arrowprops=dict(arrowstyle="->"),
        )
        ax.add_artist(ab)
        ab.set_visible(False)
        ax.figure.canvas.mpl_connect("pick_event", __overlap_onpick)

    plt.show()
    return figure, ax


def get_best_match(frames: FrameExtractor, distance_list: list, video_1, video_2):
    """
    Gets the best video match between two videos, given the FrameExtractor, the associated distance_list,
    and the two video names.
    """
    v1_indexes = frames.get_index_from_video_name(video_1)
    v2_indexes = frames.get_index_from_video_name(video_2)
    v1_dlist = [x for x in distance_list if x[1][0] in v1_indexes]
    print(video_1)
    print(video_2)
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

    return (frames.all_images[relevant_index[0]], frames.all_images[relevant_index[1]])


def show_some_results(neighbours, number_to_show=5):
    """Show some results from each video, i.e. not all possible results.

    This results in a briefer list and better overview of the results.

    Args:
        am (AnnotationMatcher): [description]
        frames (FrameExtractor): [description]
        distance_list (list): [description]
        number_to_show (int, optional): [description]. Defaults to 5.
    """
    frames = neighbours.frames

    for video_name_to_show in frames.video_list:
        video_indexes = frames.get_index_from_video_name(video_name_to_show)
        # Find the distance matching vid1_frame in the distance list
        video_dis = list(
            (x for x in neighbours.distance_list if x[1][0] in video_indexes)
        )
        # Sort from frame 0 -> last frame
        video_dis = sorted(video_dis, key=lambda x: x[1][0])
        remaining_to_show = number_to_show
        for d, i in tqdm(video_dis):
            if remaining_to_show == 0:
                break
            remaining_to_show -= 1
            # Ensure that only images from *other* video files are shown by ignoring similar results from the same file
            show_similar_videos(d, i, frames.all_images, distance_threshold=10000000)


def print_video_match_overview(
    distance_list, frames: FrameExtractor, distance_threshold=20000
):
    """Prints an overview of the videos matched within a specified distance threshold according
    to the distance list.

    The output includes total valid matches, total amount of frames and video name for each video.

    Args:
        distance_list (list): A faiss distance list
        frames ([type]): The associated FrameExtractor class
        distance_threshold (int, optional): The maximum allowed threshold for distance. Defaults to 20000.
    """
    video_list = list(frames.cached_video_index.keys())

    dlist_sorted = sorted(distance_list, key=lambda x: x[1][0])  # Sort index
    dlist_indexes = np.array([x[1][0] for x in dlist_sorted])

    for vid in video_list:
        vid_indexes = frames.get_index_from_video_name(vid)
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

        vid_dlist = dlist_sorted[vid_loc[0] : vid_loc[-1]]

        # This checks for number of matches where distance < 20000
        vid_distances = [
            (np.sum(np.array(x) < distance_threshold) - 1) for x, y in vid_dlist
        ]

        total = np.sum(vid_distances)
        # TODO: Add sorting?
        print(f"{total:10} / {len(vid_indexes):10}: {vid} ")


def fill_database_with_frames(db_file, dl_network: Network, frames: FrameExtractor):
    with dbhandler.SavedComparisonsDatabase(db_file) as dbc:
        already_exist = dbc.get_thumnbnail_list()
        for img in tqdm(frames.all_images):
            if img in already_exist:
                continue
            processed_img = process_image(img, dl_network.target_size, trim=True)
            dbc.add_thumbnail(img, processed_img)


def __calc_fun(args):
    img, target_size = args
    processed_img = process_image(img, target_size, trim=True)
    compressed_img = dbhandler.SavedComparisonsDatabase.get_thumbnail_compressed(
        processed_img
    )
    return (img, compressed_img)


def fill_db_with_thumbnails(
    dl_network: Network, frames: FrameExtractor, db_file, pool_size=6
):
    """Fill the database with thumbnails from the given frame class."""
    with Pool(pool_size) as pool:
        with dbhandler.SavedComparisonsDatabase(db_file) as dbc:
            already_exist = dbc.get_thumnbnail_list()
            tn_set = set()

            for img in tqdm(frames.all_images):
                if img in already_exist:
                    continue
                tn_set.add(img)

            new_iter = ([x, dl_network.target_size] for x in tn_set)
            for img, compressed_image in pool.imap_unordered(__calc_fun, new_iter):
                dbc.add_thumbnail(img, None, compressed_image)


# from skimage import img_as_ubyte # TODO: Check if this is actually needed or not!
# from io import BytesIO
# from PIL import Image as PILImage

# def get_frame_thumbnail_from_db(image, dbfile):
#     with dbhandler.SavedComparisonsDatabase(dbfile) as dbc:
#         data = dbc.get_thumbnail(image)
#         out = BytesIO()
#         data_ubyte = img_as_ubyte(np.divide(data.squeeze(),255))
#         img = PILImage.fromarray(data_ubyte)
#         return img


def _get_font(font_size=20):
    """Get a system font.

    Args:
        font_size (int, optional): The requested font size. Defaults to 20.

    Returns:
        The font, as a truetype
    """
    installed_fonts = findSystemFonts(fontpaths=None, fontext="ttf")
    # Change index here if required
    return PIL.ImageFont.truetype(installed_fonts[0], font_size)


def merge_frames_to_image(
    neighbours, frame_distances, frame_indexes, frame_resize=(200, 200), font_size=16
):
    frames = neighbours.frames
    frame_count = len(frame_indexes)
    frame_files = [frames.all_images[x] for x in frame_indexes]

    font = _get_font(font_size)

    merged = PIL.Image.new(
        "RGB", (frame_resize[0] * frame_count, frame_resize[1]), (250, 250, 250)
    )

    for start_pos, (distance, frame_file) in enumerate(
        zip(frame_distances, frame_files)
    ):
        img = PIL.Image.open(frame_file).resize(frame_resize)
        draw_im = PIL.ImageDraw.Draw(img)

        video_text = f"{distance:.3f}"
        draw_im.text((0, 0), video_text, (255, 255, 255), font=font)

        merged.paste(img, (frame_resize[0] * start_pos, 0))

    return merged


def get_neighbour_data(neighbours, distances, indexes, include_index=False):
    """Gets a pandas dataframe containng the distance, video name, and time
        (both second and timestamp) of all neighbours in the supplied list.

    Args:
        neighbours ([type]): The full neighbours list
        distances ([type]): The relevant distances
        indexes ([type]): The relevant neighbour indexes
        include_index (bool, optional): If true, image index is
            included in the dataframe. Defaults to False.

    Returns:
        [type]: [description]
    """
    columns = ["Distance", "Video name", "Frame (S)", "Frame (HH:MM:SS)"]

    def create_dict(x):
        return dict(zip(columns, x))

    def create_timestamp(x):
        return time.strftime("%H:%M:%S", time.gmtime(x))

    reference = indexes[0]
    vid, frame_start = neighbours.frames.get_video_and_start_time_from_index(reference)
    timestamp = create_timestamp(frame_start)

    result_list = [create_dict([distances[0], vid, frame_start, timestamp])]
    real_indexes = [reference]

    for d, i in zip(distances[1:], indexes[1:]):
        real_indexes.append(i)
        vid, frame_start = neighbours.frames.get_video_and_start_time_from_index(i)
        timestamp = create_timestamp(frame_start)
        result_list.append(create_dict([d, vid, frame_start, timestamp]))

    df = pd.DataFrame(result_list, columns=columns)

    if include_index:
        df["Index"] = real_indexes
    return df


def get_one_index_per_video(neighbours, d, i, total_results=6):
    """Ensures each video is only present once in the supplied list of indexes.

    Args:
        neighbours ([type]): The full neighbours
        d ([type]): The relevant distances
        i ([type]): The relevant indexes
        total_results (int, optional): Number of results to return. Defaults to 6.

    Returns:
        [type]: [description]
    """
    used_videos = set()
    frames = neighbours.frames
    return_d = []
    return_i = []

    for dd, ii in zip(d, i):
        video = frames.get_video_name_from_index(ii)
        if video not in used_videos:
            used_videos.add(video)
            total_results -= 1
            return_d.append(dd)
            return_i.append(ii)
            if total_results <= 0:
                break
    return (return_d, return_i)


def show_best_results_per_video(
    neighbours,
    frames_to_show_per_video=5,
    neighbours_to_show_per_frame=5,
    number_of_neighbours_to_consider=1,
    max_number_to_return=-1
):
    """Shows a number of best results per video.

    If number_of_neighbours_to_consider is set to 1, this is sorted by closest distance
    to nearest neighbour first.

    Otherwise, it is the mean of the x number of neighbours this value is set to.

    Args:
        neighbours ([type]): The set of neighbours to base the display on
        frames_to_show_per_video (int, optional): The number of separate frames to show
            from each video. Defaults to 5.
        neighbours_to_show_per_frame (int, optional): The number of neighbours to show per
            frame. Defaults to 5. Can be fewer if no more neighbours exist.
        number_of_neighbours_to_consider (int, optional): How many neighbours to consider
            when ranking, see longer description above. Defaults to 1.
    """
    dlist = neighbours.distance_list
    dlist = sorted(
        dlist, key=lambda x: np.mean(x[0][1 : number_of_neighbours_to_consider + 1])
    )

    for vid in neighbours.frames.video_list:
        print(f"Video: {vid}")
        vid_indexes = neighbours.frames.cached_video_index[vid]
        remaining_to_show = frames_to_show_per_video
        for d, i in dlist:
            if i[0] in vid_indexes:
                remaining_to_show -= 1
                max_number_to_return -= 1
                d, i = get_one_index_per_video(
                    neighbours, d, i, neighbours_to_show_per_frame + 2
                )
                df = get_neighbour_data(neighbours, d, i)
                print(df.to_string())
                display(merge_frames_to_image(neighbours, d, i))

            if remaining_to_show <= 0:
                break
            
            if max_number_to_return == 0:
                return


def _safe_filter_fun(filter_function, d, i):
    """Helper function that returns False if an exception occurs instead of stopping the loop."""
    try:
        return filter_function(d, i)
    except:
        return False


def show_samples(
    neighbours,
    samples_to_show,
    filter_function,
    neighbours_to_show=5,
    include_index=False,
):
    """Shows frames (and a number of neighbours) where the given filter_function returns True.

    This can be used to find examples where the first neighbour is within a certain interval, for example:

    show_samples(neighbours, 3, lambda d,i: d[1] > 0 and d[1] < 15000, include_index=True)

    Args:
        neighbours ([type]): The relevant neighbours
        samples_to_show ([type]): How many samples (frames) to show
        filter_function ([type]): The filter function, which takes the distance and index list (d, i) as input and returns True if it matches.
        neighbours_to_show (int, optional): How many neighbours to show, default 5
        include_index (bool, optional): Whether or not the index of the samples should be included in the dataframe. Defaults to False.
            This can be of particular interest if the user is looking for frames to exclude.
    """
    dlist = [
        (d, i)
        for d, i in neighbours.distance_list
        if _safe_filter_fun(filter_function, d, i)
    ]
    if len(dlist) == 0:
        print("No remaining samples after filter!")
        return
    samples_to_show = np.min([samples_to_show, len(dlist)])

    for curr_index in random.sample(range(len(dlist)), samples_to_show):
        d, i = dlist[curr_index]
        d, i = get_one_index_per_video(neighbours, d, i, neighbours_to_show + 2)
        df = get_neighbour_data(neighbours, d, i, include_index=include_index)
        print(df.to_string())
        display(merge_frames_to_image(neighbours, d, i))


def execute_filter(neighbours, filter_fun, *args, **kwargs):
    """Helper function to execute a filter and

    Args:
        neighbours ([type]): [description]
        filter_fun ([type]): [description]
    """
    remaining_count = neighbours._get_remaining_count()
    max_value = np.max(remaining_count)

    neighbours.plot_remaining_statistics(
        figure_size=(10, 5), binwidth=10, binrange=(0, max_value + 10)
    )
    fig = plt.gcf()

    # TODO: Simply send this instead while creating!
    titles = [
        "No. of neighbours before this filter",
        "Distance metrics before this filter",
    ]
    for ax, title in zip(fig.get_axes(), titles):
        ax.set_title(title)

    ax_lims = [(ax.get_ylim(), ax.get_xlim()) for ax in fig.get_axes()]
    #     ax_binwidths = [ax.binwidthnumber ]
    change = FilterChange(neighbours)

    filter_fun(*args, **kwargs)

    df = FilterChange.changes_in_short(change, neighbours).round(2)
    display(df)

    neighbours.plot_remaining_statistics(figure_size=(10, 5), binwidth=10)
    fig = plt.gcf()

    # TODO: Simply send this instead while creating!
    titles = [
        "No. of neighbours after this filter",
        "Distance metrics after this filter",
    ]
    for ax, title in zip(fig.get_axes(), titles):
        ax.set_title(title)

    # Normalize filter to ensure same limits; gives better understanding of how much was removed.
    for (ylim, xlim), ax in zip(ax_lims, fig.get_axes()):
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    return neighbours


# IPYAGGRID


def df_to_col_defs(df):
    """
    This is to ensure that the grid works properly by temporarily changing the name of columns,
    ensuring that there is no naming problems.
    """
    df2 = df.copy()
    col_list = []
    new_columns = []
    for i, col in enumerate(df.columns):
        new_name = f"col{i}"
        new_columns.append(new_name)
        col_list.append({"field": new_name, "headerName": col})
    df2.columns = new_columns

    return df2, col_list


def get_grid(df):
    """Creates a grid from a dataframe;
    The grid is an interactive ipyaggrid widget that allows for sorting and filtering.

    Args:
        df ([type]): The dataframe

    Returns:
        The grid widget
    """

    new_df, column_defs = df_to_col_defs(df)

    grid_options = {
        "enableColResize": True,
        "columnDefs": column_defs,
        "enableFilter": True,
        "enableSorting": True,
        "animateRows": True,
        "groupMultiAutoColumn": True,
    }

    grid1 = Grid(
        quick_filter=True,
        compress_data=False,
        grid_data=new_df,
        grid_options=grid_options,
        columns_fit="auto",
    )
    return grid1
