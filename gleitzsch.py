#!/usr/bin/env python3
"""Gleitzsch core."""
import argparse
import sys
import os
import random
import string
import subprocess
from subprocess import DEVNULL
import numpy as np
from skimage import io
from skimage import img_as_float
from skimage.util import img_as_ubyte
from skimage import transform as tf
from skimage import exposure

__author__ = "Bogdan Kirilenko, 2020"
__version__ = 4.0

# text constants
RB_SHIFT = "rb_shift"
GLITTER = "glitter"
GAMMA_CORRECTION = "gammma_correction"
ADD_TEXT = "add_text"
RANDOM = "random"
TEMP = "temp"

MIN_IM_SIZE = 64


class Gleitzsch:
    """Gleitzsch core class."""
    def __init__(self, image_path, size=0, verbose=False):
        """Init gleitzsch class."""
        # get image array
        self.verbose = verbose
        self.im_arr, _ = self.__read_im(image_path, size)
        self.lame_bin = "lame"
        self.__check_lame()  # check that lame is available
        self.supported_filters = [GLITTER, RB_SHIFT, GAMMA_CORRECTION, ADD_TEXT]
        # create temp directory
        self.tmp_dir = os.path.join(os.path.dirname(__file__), TEMP)
        os.mkdir(self.tmp_dir) if not os.path.isdir(self.tmp_dir) else None
        self.temp_files = []  # collect temp files here (to delete later)
        self.gamma = 0.4
        self.v("Gleitzsch instance initiated successfully")
    
    def __read_im(self, image_path, size):
        """Read image, return an array and shape."""
        self.v(f"Reading file {image_path}")
        self.__die(f"File {image_path} doesn't exist!") if not os.path.isfile(image_path) else None
        matrix = img_as_float(io.imread(image_path))  # np array now
        # image migth be either 3D or 2D; if 2D -> make it 3D
        if len(matrix.shape) == 3:
            pass  # it's a 3D array already
        elif len(matrix.shape) == 2:
            # monochrom image; all procedures are 3D array-oriented
            layer = np.reshape(matrix, (matrix.shape[0], matrix.shape[1], 1))
            matrix = np.concatenate((layer, layer, layer), axis=2)
        else:  # something is wrong
            self.__die("Image {image_path} is corrupted")
        # resize if this required
        if size == 0:
            # keep size as is (not recommended)
            im = matrix.copy()
            w, h, _ = im.shape
        elif size < MIN_IM_SIZE:  # what if size is negative?
            self.__die("Image size (long side) must be > 64, got {size}")
        else:
            # resize the image
            scale_k = max(matrix.shape[0], matrix.shape[1]) / size
            h = int(matrix.shape[0] / scale_k)
            w = int(matrix.shape[1] / scale_k)
            im = tf.resize(image=matrix, output_shape=(h, w))
        self.v(f"Sucessfully open {image_path}, image_shape {w}x{h}")
        return im, (w, h)

    def __check_lame(self):
        """Check that lame is installed."""
        check_cmd = f"{self.lame_bin} --version"
        rc = subprocess.call(check_cmd, shell=True, stdout=DEVNULL)
        if rc == 0:
            self.v("Lame installation detected")
        else:
            self.__die("Lame installation not found, abort")

    def apply_filters(self, filters_all):
        """Apply filters to image one-by-one."""
        self.v(f"Calling apply filters function")
        self.text_position = RANDOM

        if not filters_all:  # no filters: nothing to do
            return
        # keep available filters only + that have value
        filters = {k: v for k, v in filters_all.items()
                   if k in self.supported_filters and v}
        # better to keep them ordered
        filters_order = sorted(filters.keys(), key=lambda x: self.supported_filters.index(x))
        for filt_id in filters_order:
            value = filters[filt_id]
            self.v(f"Applying filter: {filt_id}, value={value}")
            if filt_id == RB_SHIFT:
                self.__apply_rb_shift(value)
            elif filt_id == GLITTER:
                self.__apply_glitter(value)

    def __apply_glitter(self, value):
        """Apply glitter."""
        dots = []  # fill this list with dot coordinates
        _dot_size = 3
        w, h, _ = self.im_arr.shape
        for _ in range(value):
            # just randomly select sime coords
            dx = random.choice(range(_dot_size, w - _dot_size))
            dy = random.choice(range(_dot_size, h - _dot_size))
            dots.append((dx, dy))
        for dot in dots:
            self.im_arr[dot[0] - 1: dot[0], dot[1] - 3: dot[1] + 3, :] = 1

    def __apply_rb_shift(self, value):
        """Draw chromatic abberations."""
        _init_shape = self.im_arr.shape
        # extract different channels
        red = self.im_arr[:, :, 0]
        green = self.im_arr[:, :, 1]
        blue = self.im_arr[:, :, 2]

        # resize different channels to create the effect
        # define new sizes
        red_x, red_y = _init_shape[0], _init_shape[1]
        self.v(f"Red channel size: {red_x}x{red_y}")
        green_x, green_y = _init_shape[0] - value, _init_shape[1] - value
        self.v(f"Green channel size: {green_x}x{green_y}")
        blue_x, blue_y = _init_shape[0] - 2 * value, _init_shape[1] - 2 * value
        self.v(f"Blue channel size: {blue_x}x{blue_y}")

        # check that sizes are OK
        channel_borders = (red_x, red_y, green_x, green_y, blue_x, blue_y)
        if any(x < 1 for x in channel_borders):
            self.__die(f"{RB_SHIFT} got too bit value {value}; cannot apply")

        # apply resize procedure
        red = tf.resize(red, output_shape=(red_x, red_y))
        green = tf.resize(green, output_shape=(green_x, green_y))
        blue = tf.resize(blue, output_shape=(blue_x, blue_y))

        w, h = blue.shape  # temporary shape (minimal channel size)
        self.v(f"Updated image size: {w}x{h}")
        ktd2 = int(value / 2)
        red_n = np.reshape(red[value: -value, value: -value],
                           (w, h, 1))
        green_n = np.reshape(green[ktd2: -1 * ktd2, ktd2: -1 * ktd2],
                             (w, h, 1))
        blue_n = np.reshape(blue[:, :], (w, h, 1))
        # save changes to self.im_arr
        self.im_arr = np.concatenate((red_n, green_n, blue_n), axis=2)
        # reshape it back
        self.im_arr = tf.resize(self.im_arr , (_init_shape[0], _init_shape[1]))
        self.v(f"Sucessfully applied {RB_SHIFT} filter")

    def mp3_compression(self, attrs):
        """Compress and decompress the image using mp3 algorithm."""
        # split image in channels
        self.v("Applying mp3 compression")
        # apply gamma correction upfront
        self.im_arr = exposure.adjust_gamma(image=self.im_arr, gain=self.gamma)
        w, h, _ = self.im_arr.shape
        shift_size = h // 3
        red = self.im_arr[:, :, 0]
        green = self.im_arr[:, :, 1]
        blue = self.im_arr[:, :, 2]
        channels = (red, green, blue)
        gliched_channels = []
        # process them separately
        for channel in channels:
            # need 1D array now
            orig_size = w * h
            channel_flat = np.reshape(channel, newshape=(orig_size, ))
            int_form_nd = np.around(channel_flat * 255, decimals=0)
            int_form_nd[int_form_nd > 255] = 255
            int_form_nd[int_form_nd < 0] = 0
            # convert to bytes
            int_form = list(map(int, int_form_nd))
            bytes_str = bytes(int_form)

            # define temp file paths
            raw_chan_ = os.path.join(self.tmp_dir, f"{self.__id_gen()}.blob")
            mp3_compr_ = os.path.join(self.tmp_dir, f"{self.__id_gen()}.mp3")
            mp3_decompr_ = os.path.join(self.tmp_dir, f"{self.__id_gen()}.mp3")
            # save paths (to remove the files later)
            self.temp_files.extend([raw_chan_, mp3_compr_, mp3_decompr_])

            # save bytes so a pseudo-wav file
            self.v(f"Bytes size before compression: {orig_size}")
            with open(raw_chan_, "wb") as f:
                f.write(bytes_str)
            
            # define compress-decompress commands
            mp3_compr_cmd = f'{self.lame_bin} -r --unsigned -s 16 -q 8 --resample 16 ' \
                            f'--bitwidth 8 -b 16 -m m {raw_chan_} "{mp3_compr_}"'
            mp3_decompr_cmd = f'{self.lame_bin} --decode -x -t "{mp3_compr_}" {mp3_decompr_}'

            # call compress-decompress commands
            self.__call_proc(mp3_compr_cmd)
            self.__call_proc(mp3_decompr_cmd)

            # read decompressed file | get raw sequence
            with open(mp3_decompr_, "rb") as f:
                mp3_bytes = f.read()
            upd_size = len(mp3_bytes)
            self.v(f"Bytes size after compression: {upd_size}")
            # usually array size after compression is bigger
            proportion = upd_size // orig_size
            # split in chunks of proportion size, take the 1st element from each
            bytes_num = len(bytes_str) * proportion
            decompressed = mp3_bytes[:bytes_num]
            glitched_channel = np.array([pair[0] / 255 for pair
                                         in self.parts(decompressed, proportion)])
            glitched_channel = np.reshape(glitched_channel, newshape=(w, h, 1))
            gliched_channels.append(glitched_channel)
        self.v("Concatenation of the mp3d image + rolling + adjust contrast")
        self.im_arr = np.concatenate(gliched_channels, axis=2)
        self.im_arr  = np.roll(a=self.im_arr, axis=1, shift=shift_size)
        perc_left, perc_right = np.percentile(self.im_arr, (5, 95))
        self.im_arr = exposure.rescale_intensity(self.im_arr, in_range=(perc_left, perc_right))
        self.__remove_temp_files()
    
    def __call_proc(self, command):
        """Call command using subprocess."""
        self.v(f"Calling command: {command}")
        rc = subprocess.call(command, shell=True, stderr=DEVNULL)
        if rc != 0:
            self.__die(f"Error! Command {command} died!")

    def __remove_temp_files(self):
        """Remove temp files listed in the self.temp_files."""
        self.v(f"Removing temp files: {self.temp_files}")
        for tmp_file in self.temp_files:
            os.remove(tmp_file) if os.path.isfile(tmp_file) else None

    def save(self, path):
        """Save the resulting image."""
        self.v(f"Saving image to: {path}")
        io.imsave(fname=path, arr=img_as_ubyte(self.im_arr))

    def v(self, msg):
        """Show verbose message."""
        sys.stderr.write(f"{msg}\n") if self.verbose else None

    @staticmethod
    def __die(message, rc=1):
        """Write message and quit."""
        sys.stderr.write("Error!\n")
        sys.stderr.write(f"{message}\n")
        sys.exit(rc)
    
    @staticmethod
    def __id_gen(size=12, chars=string.ascii_uppercase + string.digits):
        """Return random string for temp files."""
        return "".join(random.choice(chars) for _ in range(size))
    
    @staticmethod
    def parts(lst, n):
        """Split an iterable into a list of lists of len n."""
        return [lst[i:i + n] for i in iter(range(0, len(lst), n))]


def parse_args():
    """Parse cmd args."""
    app = argparse.ArgumentParser()
    app.add_argument("input", help="Input image")
    app.add_argument("output", help="Output image")
    app.add_argument("--size", default=1000, type=int, help="Image size (long side)")
    app.add_argument("--verbose", "-v", action="store_true", dest="verbose",
                     help="Verbosity mode on.")
    # filters
    app.add_argument("--rb_shift", "-r", default=0, type=int,
                     help="RGB abberations, the bigger value -> the higher intensivity")
    app.add_argument("--glitter", "-g", default=0, type=int,
                     help="Add glitter, The bigger value -> the bigger sparks")
    app.add_argument("--add_text", "-t", default=None,
                     help="Add text (position is random)")
    app.add_argument("--text_position", "--tp", type=str,
                     help="Pre-define text coordinates (left corner) "
                          "two comma-separated values like 100,50")
    if len(sys.argv) < 3:
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    gleitzsch = Gleitzsch(args.input, args.size, args.verbose)
    gleitzsch.apply_filters(vars(args))  # as a dict: filter id -> value
    gleitzsch.mp3_compression(vars(args))
    gleitzsch.save(args.output)
