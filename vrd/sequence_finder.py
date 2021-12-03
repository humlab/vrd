"""A low-memoryt variant of sequence finder 

"""
from copy import deepcopy
from typing import List

import numpy as np

from .neighbours import Neighbours


class SequenceFinder:
    """Testing a low-memory approach to finding sequences.

    Returns:
        A sequence finder object
    """

    neighbours: Neighbours
    found_sequences: dict
    allowed_sequence_gap: int
    minimum_sequence_length: int
    sequence_list: list

    def __init__(
        self,
        neighbours: Neighbours,
        maximum_distance: int,
        minimum_sequence_length=5,
        allowed_sequence_gap=0,
    ):
        self.neighbours = deepcopy(neighbours)
        self.minimum_sequence_length = minimum_sequence_length
        self.allowed_sequence_gap = allowed_sequence_gap
        self.sequence_list = []

        # Check if we have some issues...
        num_unique = len(np.unique([i[0] for d, i in neighbours.distance_list]))
        num_actual = len(neighbours.distance_list)
        if num_unique != num_actual:
            print(
                f"Number of unique / actual numbers do not match. Actual: {num_actual}, unique: {num_unique}"
            )
            dlist = self.neighbours.distance_list
            #  Remove duplicates if present.
            new_dlist = [dlist[0]]
            last_index = new_dlist[0][1][0]
            for d, i in dlist[1:]:
                if i[0] == last_index:
                    continue
                last_index = i[0]
                new_dlist.append((d, i))
            self.neighbours.distance_list = new_dlist

    def find_sequences(self):
        """Finds the sequences from the neighbours.
        We assume distances have been fixed already,
        i.e. that any distance from A->B is also in B->A.
        """
        videos = deepcopy(self.neighbours.frames.video_list)
        sequence_dlist = deepcopy(self.neighbours.distance_list)

        while len(videos) > 0:
            vid1 = videos.pop()
            vid1_indexes = self.neighbours.frames.get_index_from_video_name(vid1)

            # Get all distances for this video and sort them in chronological order (by index)
            vid1_dlist = [(d, i) for d, i in sequence_dlist if i[0] in vid1_indexes]
            vid1_dlist = sorted(vid1_dlist, key=lambda x: x[1][0])

            # vid1_index_sets = [set(i) for d, i in vid1_dlist]

            ignore_list: List[set] = [set()]

            for dlist_index, (d, i) in enumerate(vid1_dlist):
                ignore_indexes = ignore_list.pop()

                for _d, _i in zip(d[1:], i[1:]):
                    if _i in ignore_indexes:
                        continue
                    # seq = this,
                    # _d,_i = distance_lists of single frame in vid1 (vid1_dlist)
                    # vid1_dlist = full vid1_dlist
                    # dlist_index = index fod _d, _i in vid1_dlist
                    sequence = self.is_sequence(
                        _d, _i, vid1_dlist[dlist_index:], dlist_index
                    )
                    if sequence is not None:
                        # Add the sequence to the ignore list so we avoid finding the same sequence many times.

                        for ignore_index, (v1_i, distance, v2_i) in enumerate(sequence):
                            if ignore_index > len(ignore_list):
                                ignore_list.append(set())
                            ignore_list[ignore_index] = ignore_list[ignore_index].add(
                                v2_i
                            )

                        self.sequence_list.append(sequence)

            # TODO: REMOVE ALL INSTANCES OF vid1_indexes FROM DISTANCE LIST!

    #         new_dlist = list()
    #         for d,i in sequence_dlist:

    #         break # REMOVE THIS TO CHECK ALL VIDEOS

    def is_sequence(self, start_dist, start_index, vid_dlist, dlist_index):
        """Determine if a certain spot is the start of a sequence

        Args:
            start_dist ([type]): [description]
            start_index ([type]): [description]
            vid_dlist ([type]): [description]
            dlist_index ([type]): [description]
        """
        # dlist is trimmed video list (i.e. starts where this frame is)
        orig_dlist_index = dlist_index
        allowed_gap = self.allowed_sequence_gap
        min_len = self.minimum_sequence_length

        sequence = [(dlist_index, start_dist, start_index)]

        curr_index_to_find = start_index + 1
        curr_gap = 0
        last_orig_index = vid_dlist[0][1][0]  # This assumes dlist is trimmed!
        for curr_index, (d, i) in enumerate(vid_dlist[1:]):  # skip first...
            curr_gap += i[0] - last_orig_index

            last_orig_index = i[0]
            if (curr_gap - 1) > allowed_gap:
                #             print(f'Exiting after {curr_gap}')
                break

            found_d, found_i = next(
                ((_d, _i) for _d, _i in zip(d, i) if _i == curr_index_to_find),
                (None, None),
            )

            if found_i is not None:
                #             print('found i')
                curr_index_to_find += 1
                sequence.append((dlist_index + curr_index + 1, found_d, found_i))
                curr_gap = 0

        if len(sequence) >= self.minimum_sequence_length:
            # TODO: REMOVE THIS SEQUENCE FROM THE DISTANCE LIST TO AVOID FINDING SAME SEQUENCE MANY TIMES
            return sequence  
        return None
