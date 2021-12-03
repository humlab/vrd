"""Allows the user to view sequences by duration, in order of lowest distance"""

from typing import Callable, Tuple

import numpy as np
import pandas as pd
from IPython.display import DisplayHandle, clear_output, display
from ipywidgets import widgets
from ipywidgets.widgets.widget_box import HBox
from ipywidgets.widgets.widget_selection import Dropdown
from ipywidgets.widgets.widget_string import Label
from pandas.core.frame import DataFrame
from tqdm.auto import tqdm

from ..overlap_calculator import OverlapCalculator


class SequencesWithLowestDistancesByDuration(widgets.DOMWidget):
    """Creates a widget that allows the user to view sequences as images,
    selected by their duration."""

    df: DataFrame
    oc: OverlapCalculator
    df_handle: DisplayHandle = None
    images_handle: widgets.Output = None
    sequence_drop: Dropdown = None
    show_button: widgets.Button = None
    progress_label: Label = None
    score_function: Callable = None
    image_resize: Tuple[int, int]

    def __init__(
        self,
        oc: OverlapCalculator,
        score_function: Callable = np.median,
        image_resize: Tuple[int, int] = (200, 200),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.oc = oc
        self.df = oc.get_all_sequences(score_function=score_function)
        self.score_function = score_function
        self.image_resize = image_resize

        self.df_handle = DisplayHandle()
        self.images_handle = widgets.Output()
        self.sequence_drop = Dropdown(options=np.unique(self.df["Duration"]))
        self.show_button = widgets.Button(description="Show")
        self.progress_label = Label("Idle")

        self.show_button.on_click(self.sequence_show_button_clicked)
        self.sequence_drop.observe(self.sequence_length_change, names="value")

        display(
            HBox(
                [
                    Label("Sequence length: "),
                    self.sequence_drop,
                    self.show_button,
                    self.progress_label,
                ]
            )
        )
        self.df_handle.display("Choose squence")
        display(HBox([self.images_handle]))

    def sequence_length_change(self, change):
        if change["name"] == "value" and (change["new"] != change["old"]):
            df = self.df
            sequences = df["Duration"] == self.sequence_drop.value
            self.df_handle.display(df.loc[sequences])
            with self.images_handle:
                clear_output()
            self.progress_label.value = "Idle"

    def sequence_show_button_clicked(self, b):
        with self.images_handle:
            self.progress_label.value = "Working..."
            df = self.df
            relevant_sequences = df.loc[df["Duration"] == self.sequence_drop.value]
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
            self.progress_label.value = "Finished"
