"""Widget for 'manual', i.e. not sorted, preview of remaining frames in a neighbours class.
"""

import ipywidgets as widgets
import numpy as np
import PIL
from IPython.display import HTML, DisplayHandle, Image, clear_output, display
from ipywidgets import Dropdown, HBox, Label
from PIL import ImageOps

from ..image_preprocessing import process_image
from ..neighbours import Neighbours
from ..notebook_helper import (
    get_neighbour_data,
    get_one_index_per_video,
    merge_frames_to_image,
)


class ManualImagePreview(widgets.DOMWidget):
    neighbours: Neighbours
    vid_drop: Dropdown
    frame_drop: Dropdown
    frame_count_label: Label
    statistics_handle: DisplayHandle
    image_handle: DisplayHandle
    show_debug_handle: widgets.Output
    network_target_size: int
    number_of_neighbours_shown: int

    def __init__(
        self,
        neighbours: Neighbours,
        network_target_size: int,
        number_of_neighbours_shown: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.neighbours = neighbours
        self.network_target_size = network_target_size
        self.number_of_neighbours_shown = number_of_neighbours_shown

        self.vid_drop = Dropdown(options=neighbours.frames.video_list)
        self.frame_drop = Dropdown(options=["0"])

        self.frame_count_label = Label("Total: N/A")
        self.statistics_handle = DisplayHandle()

        self.image_handle = DisplayHandle()
        self.show_debug_handle = widgets.Output()

        self.update_frame_count()
        self.vid_drop.observe(self.video_change, names="value")
        self.frame_drop.observe(self.frame_change, names="value")

        self.show_debug = widgets.Button(description="Show debug image")
        self.show_debug.on_click(self.show_debug_button_clicked)

        self.statistics_handle.display("No statistics yet")

        display(
            HBox(
                [
                    Label("Video: "),
                    self.vid_drop,
                    Label("   Frame:"),
                    self.frame_drop,
                    self.frame_count_label,
                    self.show_debug,
                ]
            )
        )
        self.image_handle.display("None yet selected")
        display(HBox([self.show_debug_handle]))

    def update_frame_count(self):
        frames = list(self.neighbours.frames.cached_video_index[self.vid_drop.value])
        frame_count = len(frames)
        self.frame_drop.options = [f"{i}" for i in range(frame_count)]
        self.frame_count_label.value = f"Video frame count: {frame_count}"

    def video_change(self, change):
        if change["name"] == "value" and (change["new"] != change["old"]):
            self.update_frame_count()

    def frame_change(self, change):
        if change["name"] == "value" and (change["new"] != change["old"]):
            with self.show_debug_handle:
                clear_output()
            frames = list(
                self.neighbours.frames.cached_video_index[self.vid_drop.value]
            )
            frame = int(self.frame_drop.value)
            actual_frame_index = frames[frame]
            dlist = self.neighbours.distance_list
            match = [(d, i) for d, i in dlist if i[0] == actual_frame_index]
            if len(match) > 0:
                d, i = match[0]
                d, i = get_one_index_per_video(
                    self.neighbours, d, i, self.number_of_neighbours_shown
                )
                df = get_neighbour_data(self.neighbours, d, i)
                self.statistics_handle.display(HTML(df.to_html()))
                self.image_handle.display(merge_frames_to_image(self.neighbours, d, i))
            else:
                self.statistics_handle.display("Frame has been filtered")
                self.image_handle.display("")

    @staticmethod
    def get_debug_image(neighbours, video, frame, target_size):
        """Gets a debug image, i.e. the image as input into the neural network,
        to check if the image processing is the original cause of invalid results."""
        frames = neighbours.frames
        img = frames.all_images[sorted(list(frames.cached_video_index[video]))[frame]]
        processed_img = process_image(img, target_size, trim=True)
        fixed_img = (processed_img.squeeze() * 255).astype(np.uint8)

        # We could probably fix the inverted image earlier, but this works too.
        pil_image = ImageOps.invert(
            PIL.Image.fromarray(fixed_img[..., [0, 1, 2]].copy())
        )
        return pil_image, img

    def show_debug_button_clicked(self, b):
        """Runs when debug button is clicked, shows original and processed image!"""
        with self.show_debug_handle:
            processed_image, frame = self.get_debug_image(
                self.neighbours,
                self.vid_drop.value,
                int(self.frame_drop.value),
                self.network_target_size,
            )
            display(Image(filename=frame))
            display(processed_image)
