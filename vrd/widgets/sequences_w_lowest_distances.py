"""Includes widget for showing sequences with lowest distances between images"""
from typing import Callable

import numpy as np
import pandas as pd
from IPython.core.display import HTML, DisplayHandle, clear_output, display
from ipywidgets import Dropdown, HBox, Label, widgets
from pandas import DataFrame
from tqdm.auto import tqdm

from ..neighbours import Neighbours
from ..overlap_calculator import OverlapCalculator


class SequencesWithLowestDistances(widgets.DOMWidget):
    """Widget for showing sequences (per video) with the lowest distances.

    Args:
        widgets ([type]): [description]

    Returns:
        [type]: [description]
    """

    df_handle: DisplayHandle
    sequence_handle: widgets.Output
    sequence_drop: Dropdown
    show_button: widgets.Button
    video_sequence_progress_label: Label

    oc: OverlapCalculator
    df: DataFrame
    neighbours: Neighbours
    score_function: Callable

    def __init__(self, oc: OverlapCalculator, score_function=np.median) -> None:
        super().__init__()
        self.oc = oc
        self.score_function = score_function

        self.df = oc.get_all_sequences(score_function=self.score_function)
        self.df_handle = DisplayHandle()
        self.sequence_handle = widgets.Output()
        self.sequence_drop = Dropdown(options=oc.videos)
        self.show_button = widgets.Button(description="Show")

        self.sequence_drop.observe(self.sequence_video_changed, names="value")
        self.show_button.on_click(self.video_sequence_button_clicked)

        self.video_sequence_progress_label = Label("Idle")
        display(
            HBox(
                [
                    Label("Video name: "),
                    self.sequence_drop,
                    self.show_button,
                    self.video_sequence_progress_label,
                ]
            )
        )
        self.df_handle.display("Choose video")

        display(HBox([self.sequence_handle]))

    @staticmethod
    def get_relevant_sequence_df(df, video):
        """Gets sequences where the specified video is either Video 1 or Video 2"""
        return df[((df["Video 1"] == video) | (df["Video 2"] == video))]

    def sequence_video_changed(self, change):
        """Actions taken when the video selection has changed."""
        if change["name"] == "value" and (change["new"] != change["old"]):
            self.video_sequence_progress_label.value = "Working..."
            df = self.oc.get_all_sequences(score_function=self.score_function)
            sequences = self.get_relevant_sequence_df(df, self.sequence_drop.value)
            self.df_handle.display(HTML(sequences.to_html()))
            with self.sequence_handle:
                clear_output()
            self.video_sequence_progress_label.value = "Idle"

    def video_sequence_button_clicked(self, b):
        """Actually displays all sequences in the selected video"""
        with self.sequence_handle:
            self.video_sequence_progress_label.value = "Working..."

            relevant_sequences = self.get_relevant_sequence_df(
                self.oc.get_all_sequences(score_function=self.score_function),
                self.sequence_drop.value,
            )
            clear_output()
            for sequence_dict in tqdm(relevant_sequences.to_dict(orient="records")):
                vid_1, vid_2, vid_1_s, vid_2_s, duration = [
                    sequence_dict[x]
                    for x in [
                        "Video 1",
                        "Video 2",
                        "Video 1 frame (S)",
                        "Video 2 frame (S)",
                        "Duration",
                    ]
                ]

                def name_to_index(vidname):
                    for i, vid in enumerate(self.oc.neighbours.frames.video_list):
                        if vid == vidname:
                            return i
                    return None

                img = self.oc.create_sequence_comparison_image(
                    "",
                    name_to_index(vid_1),
                    name_to_index(vid_2),
                    vid_1_s,
                    vid_2_s,
                    int(duration),
                    image_resize=(200, 200),
                    font_size=14,
                )
                print(
                    pd.DataFrame()
                    .append(sequence_dict, ignore_index=True)
                    .to_string(index=False)
                )
                display(img)
            self.video_sequence_progress_label.value = "Finished"
