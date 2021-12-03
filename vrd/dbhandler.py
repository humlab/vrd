import hashlib
import io
import sqlite3
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from .image_preprocessing import process_image


class VRDDatabase:
    """A database containing the extracted fingerprints of a specific frame.

    The database is based on sqlite3.

    Usually used with the [with] keyword to ensure it is properly exited.

    Returns:
        SavedComparisonDatabase: The comparison database handle
    """

    database_file: str
    cursor: sqlite3.Cursor
    connection: sqlite3.Connection

    def __init__(self, database_file: str) -> None:
        """Initialize comparison database.
        If database already exists, will load it; otherwise, it will be created.

        Args:
            database_file (str): Path to sqlite3 database file
        """

        self.connection = sqlite3.connect(database_file)
        self.cursor = self.connection.cursor()
        self.initialize_db()

    def __enter__(self):
        """Enter function

        Returns:
            [type]: [description]
        """
        return self

    def __exit__(self, _type, _value, _traceback):
        """Exit function

        Args:
            _type ([type]): [description]
            _value ([type]): [description]
            _traceback ([type]): [description]
        """
        self.commit()
        self.close()

    def initialize_db(self):
        """Create the necessary tables and set index"""
        cursor = self.cursor
        conn = self.connection

        if not self.table_exists("saved_dl_fingerprints"):
            cursor.execute(
                """CREATE TABLE saved_dl_fingerprints (hash text, layer text)"""
            )
            cursor.execute(
                "CREATE UNIQUE INDEX fingerprint_index ON saved_dl_fingerprints(hash)"
            )
            conn.commit()
        if not self.table_exists("processed_thumbnails"):
            cursor.execute(
                """CREATE TABLE processed_thumbnails (hash text NOT NULL, thumbnail blob NOT NULL)"""
            )
            cursor.execute(
                "CREATE UNIQUE INDEX thumbnail_index ON processed_thumbnails(hash)"
            )
            conn.commit()

    def table_exists(self, tablename: str):
        """
        Returns true of the given table exists, false otherwise.
        """
        cursor = self.cursor
        self.cursor.execute(
            """SELECT count(name) FROM sqlite_master WHERE type='table' AND name=? """,
            (tablename,),
        )

        # if the count is 1, then table exists
        if cursor.fetchone()[0] == 1:
            return True
        return False

    def get_frame_list(self):
        """
        Get list of frames with hashes in database
        """
        cursor = self.cursor
        cursor.execute("""SELECT hash FROM processed_thumbnails""")
        return [x[0] for x in cursor.fetchall()]

    @staticmethod
    def hash_string(string_to_hash: str) -> str:
        """Calculate md5 hash of string

        Args:
            string_to_hash (str): The string to hash

        Returns:
            str: The md5 hash of the input string
        """
        return (hashlib.md5(string_to_hash)).hexdigest()

    def add_processed_frame(
        self, file_name_hash: str, frame: np.array, processed_frame=None
    ):
        """Add the frame image for the given hash.

        This is the image actually analyzed by the neural network.

        Args:
            file_name_hash (str): The hash representing the file name
            frame (np.array): An np-array corresponding to the frame
        """
        cursor = self.cursor
        if processed_frame is None:
            processed_frame = VRDDatabase.get_processed_frame(frame)
        cursor.execute(
            "REPLACE INTO processed_thumbnails VALUES (?,?)",
            (file_name_hash, processed_frame),
        )

    @staticmethod
    def get_processed_frame(frame: np.array):
        """Processes a frame (compresses) and returns the results.

        Args:
            frame (np.array): The frame as a numpy array

        Returns:
            The compressed frame (in byte format)
        """
        output = io.BytesIO()
        np.savez_compressed(output, x=frame)
        return output.getvalue()

    def add_layer_data(self, file_name_hash: str, layer: np.array):
        """Add the fingerprint values for the given hash.

        The fingerprint is generally from a CNN layer - therefore the name -,
        but could be any numpy array.

        Args:
            file_name_hash (str): The hash representing the file name
            layer (np.array): An np-array corresponding
        """
        cursor = self.cursor
        output = io.BytesIO()
        np.savez(output, x=layer)
        np_string = output.getvalue()
        cursor.execute(
            "REPLACE INTO saved_dl_fingerprints VALUES (?,?)",
            (file_name_hash, np_string),
        )

    def add_many_layer_data(self, hash_layer_list):
        """Add several layers at once, which improves execution speed.

        Args:
            hash_layer_list (bool): A list of tuples, containing hashes and values.
        """
        # raise NotImplementedError

        cursor = self.cursor
        # TODO: convert secondary value correcty before executing!
        cursor.executemany(
            "INSERT INTO saved_dl_fingerprints VALUES (?, ?)", hash_layer_list
        )

    def get_frame(self, file_name_hash: str):
        """Gets the compressed frame for the specified hash, if any.

        Args:
            file_name_hash (str): A file hash representing the requested frame

        Returns:
            [type]: None if it doesn't exist, otherwise a numpy array representing the previously saved layer
        """
        cursor = self.cursor
        try:
            processed_frame = next(
                cursor.execute(
                    "SELECT * FROM processed_thumbnails WHERE hash=?", (file_name_hash,)
                )
            )[1]
        except:
            return None
        data = np.load(io.BytesIO(processed_frame))

        return data["x"]

    def get_layer_data(self, file_name_hash: str):
        """Gets the layer data for the specified hash, if any.

        Args:
            file_name_hash (str): A file hash representing the requested frame

        Returns:
            [type]: None if it doesn't exist, otherwise a numpy array representing the previously saved layer
        """
        cursor = self.cursor
        try:
            layer_data = next(
                cursor.execute(
                    "SELECT * FROM saved_dl_fingerprints WHERE hash=?",
                    (file_name_hash,),
                )
            )[1]
        except:
            return None
        data = np.load(io.BytesIO(layer_data))

        return data["x"]

    def commit(self):
        """Commits to the database"""
        self.connection.commit()

    def close(self):
        """Closes the database"""
        try:
            self.connection.close()
        except:
            pass

    def fill_with_processed_frames(self, frames, neural_network, pool_size=6):
        """Fills database with previously processed frames

        Args:
            frames ([type]): [description]
            neural_network ([type]): [description]
            pool_size (int, optional): [description]. Defaults to 6.
        """
        pool = Pool(pool_size)

        already_exist = self.get_frame_list()
        tn_set = set()

        for img in tqdm(frames.all_images):
            if img in already_exist:
                continue
            tn_set.add(img)

        new_iter = ([x, neural_network.target_size] for x in tn_set)
        for img, compressed_image in pool.imap_unordered(__calc_fun, new_iter):
            self.add_processed_frame(img, None, compressed_image)


@staticmethod
def __calc_fun(args):
    img, target_size = args
    # TODO: Verify these settings, perhaps we need to grab them from elsewhere? I.e. no trim etc
    processed_img = process_image(img, target_size, trim=True)
    compressed_img = VRDDatabase.get_processed_frame(processed_img)
    return (img, compressed_img)
