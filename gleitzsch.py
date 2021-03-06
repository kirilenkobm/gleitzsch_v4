#!/usr/bin/env python3
"""Gleitzsch core."""
import argparse
import sys
import os
import random
import string
import subprocess
from subprocess import DEVNULL
# from subprocess import PIPE
from array import array
import numpy as np
from skimage import io
from skimage import img_as_float
from skimage.util import img_as_ubyte
from skimage import transform as tf
from skimage import exposure
from skimage import util
from skimage import color
from skimage import filters
try:
    from pydub import AudioSegment
except ImportError:
    sys.stderr.write("Warning! Could not import pydub.\n")
    sys.stderr.write("This library is not  mandatory, however\n")
    sys.stderr.write("filters on sound data would be not available\n")
    AudioSegment = None

__author__ = "Bogdan Kirilenko, 2020"
__version__ = 4.0

# text constants
RB_SHIFT = "rb_shift"
GLITTER = "glitter"
GAMMA_CORRECTION = "gamma_correction"
ADD_TEXT = "add_text"
VERT_STREAKS = "vert_streaks"

ADD_NOISE = "add_noise"
SOUND_QUALITY = "sound_quality"
BITRATE = "bitrate"
INTENSIFY = "intensify"
GLITCH_SOUND = "glitch_sound"
ADD_RAINBOW = "add_rainbow"
SHIFT_SIZE = "shift_size"
STRETCHING = "stretching"

RANDOM = "random"
TEMP = "temp"

MIN_IM_SIZE = 64


class Gleitzsch:
    """Gleitzsch core class."""
    def __init__(self, image_in, size=0, verbose=False):
        """Init gleitzsch class."""
        # get image array
        self.verbose = verbose
        self.im_arr, _ = self.__read_im(image_in, size)
        self.lame_bin = "lame"
        self.__check_lame()  # check that lame is available
        self.supported_filters = [GLITTER, RB_SHIFT,
                                  VERT_STREAKS, ADD_TEXT,
                                  ADD_RAINBOW, STRETCHING]
        # create temp directory
        self.tmp_dir = os.path.join(os.path.dirname(__file__), TEMP)
        os.mkdir(self.tmp_dir) if not os.path.isdir(self.tmp_dir) else None
        self.temp_files = []  # collect temp files here (to delete later)
        self.gamma = 0.4
        self.text_position = RANDOM
        self.v("Gleitzsch instance initiated successfully")

    def __read_im(self, image_in, size):
        """Read image, return an array and shape."""
        if isinstance(image_in, str) and os.path.isfile(image_in):
            self.v(f"Reading file {image_in}")
            matrix = img_as_float(io.imread(image_in))
        elif isinstance(image_in, np.ndarray):
            self.v("Reading np array")
            matrix = img_as_float(image_in)
        else:
            matrix = None
            self.__die(f"Cannot read:\n{image_in}")
        # image might be either 3D or 2D; if 2D -> make it 3D
        if len(matrix.shape) == 3:
            pass  # it's a 3D array already
        elif len(matrix.shape) == 2:
            # monochrome image; all procedures are 3D array-oriented
            layer = np.reshape(matrix, (matrix.shape[0], matrix.shape[1], 1))
            matrix = np.concatenate((layer, layer, layer), axis=2)
        else:  # something is wrong
            self.__die("Image is corrupted")
        # resize if this required
        if size == 0:
            # keep size as is (not recommended)
            im = matrix.copy()
            w, h, _ = im.shape
        elif size < MIN_IM_SIZE:  # what if size is negative?
            im, w, h = None, 0, 0
            self.__die("Image size (long side) must be > 64, got {size}")
        else:
            # resize the image
            scale_k = max(matrix.shape[0], matrix.shape[1]) / size
            h = int(matrix.shape[0] / scale_k)
            w = int(matrix.shape[1] / scale_k)
            im = tf.resize(image=matrix, output_shape=(h, w))
        self.v(f"Successfully read image; shape: {w}x{h}")
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
        """Apply filters to image one-by-one.
        
        filters_all -> a dict with filter_id: parameter.
        """
        self.v(f"Calling apply filters function")
        self.text_position = RANDOM

        if not filters_all:  # no filters: nothing to do
            return
        # keep available filters only + that have value
        filters_ = {k: v for k, v in filters_all.items()
                    if k in self.supported_filters and v}
        # better to keep them ordered
        filters_order = sorted(filters_.keys(), key=lambda x: self.supported_filters.index(x))
        for filt_id in filters_order:
            value = filters_[filt_id]
            self.v(f"Applying filter: {filt_id}, value={value}")
            if filt_id == RB_SHIFT:
                self.__apply_rb_shift(value)
            elif filt_id == GLITTER:
                self.__apply_glitter(value)
            elif filt_id == VERT_STREAKS:
                self.__add_vert_streaks()
            elif filt_id == ADD_RAINBOW:
                self.__add_rainbow()
            elif filt_id == STRETCHING:
                self.__apply_stretching()

    def __apply_stretching(self):
        """Apply stretching filter."""
        h = self.im_arr.shape[0]
        w = self.im_arr.shape[1]
        # split in 10 parts -> redefine this later
        strips_num = 10
        max_kt = 5
        strip_w = w // strips_num
        strips = []
        scales = [x for x in max_kt * np.random.sample(strips_num)]

        for j in range(strips_num):
            strip = self.im_arr[:, j * strip_w: (j + 1) * strip_w, :]
            new_shape = (h, int(strip_w * scales[j]))
            strip_res = tf.resize(strip, new_shape)
            strips.append(strip_res)
        concatenation = np.concatenate(strips, axis=1)
        self.im_arr = tf.resize(concatenation, (h, w))

    def __make_bw_(self, thr):
        """Make BW image version."""
        self.v("Producing BW version")
        col_sum = np.sum(self.im_arr, axis=2)  # sum over col channel
        bw_im = np.zeros((col_sum.shape[0], col_sum.shape[1]))
        # fill zero arr with 1 where color sum is > threshold: white
        bw_im[col_sum > thr] = 1
        # at prev step we created 2D arr, need 3D
        bw_im = np.reshape(bw_im, (bw_im.shape[0], bw_im.shape[1], 1))
        # io.imsave("test.jpg", bw_im)
        return np.concatenate((bw_im, bw_im, bw_im), axis=2)

    @staticmethod
    def __rm_bright_zones(img):
        """Remove bright zones."""
        col_sum = np.sum(img, axis=2)
        over_thr = np.reshape((col_sum - 2), (img.shape[0], img.shape[1], 1))
        over_thr = np.concatenate((over_thr, over_thr, over_thr), axis=2)
        new_im = np.where(over_thr > 0, img - over_thr, img)
        new_im[new_im < 0.0] = 0.0
        new_im[new_im > 1.0] = 1.0
        return new_im

    def __rainbow_layer_(self, bw_im):
        """Create a rainbow layer."""
        self.v("Making rainbow layer.")
        rainbow_arr = self.__apply_rb_shift(80, non_self_pic=bw_im)
        rainbow_arr = filters.gaussian(rainbow_arr,
                                       sigma=30,
                                       multichannel=True,
                                       mode='reflect',
                                       cval=0.6)
        img_hsv = color.rgb2hsv(rainbow_arr)
        img_hsv[..., 1] *= 3  # S
        img_hsv[..., 2] *= 1.4  # V
        img_hsv[img_hsv >= 1.0] = 1.0
        rainbow_arr = color.hsv2rgb(img_hsv)
        rainbow_arr = self.__rm_bright_zones(rainbow_arr)
        # io.imsave("test.jpg", rainbow_arr)
        return rainbow_arr

    def __add_rainbow(self):
        """Add rainbow to the image."""
        self.v("Adding a rainbow")
        # detect bright parts
        bw_version = self.__make_bw_(thr=2.1)
        rainbow_pic = self.__rainbow_layer_(bw_version)
        rainbow_pic /= 2
        self.im_arr = rainbow_pic + self.im_arr
        self.im_arr[self.im_arr > 1.0] = 1.0

    def __add_vert_streaks(self):
        """Add vertical streaks."""
        w, h, d = self.im_arr.shape
        processed = []
        streaks_borders_num = random.choice(range(8, 16, 2))
        streaks_borders = [0] + list(sorted(np.random.choice(range(h),
                                                             streaks_borders_num,
                                                             replace=False))) + [h]
        for num, border in enumerate(streaks_borders[1:]):
            prev_border = streaks_borders[num]
            pic_piece = self.im_arr[:, prev_border: border, :]
            if num % 2 != 0:  # don't touch this part
                processed.append(pic_piece)
                continue
            piece_h, piece_w, _ = pic_piece.shape
            piece_rearranged = []
            shifts_raw = sorted([x if x > 0 else -x for x in
                                 map(int, np.random.normal(5, 10, piece_w))])
            shifts_add = np.random.choice(range(-5, 2), piece_w)
            shifts_mod = [shifts_raw[x] + shifts_add[x] for x in range(piece_w)]
            shifts_left = [shifts_mod[x] for x in range(0, piece_w, 2)]
            shifts_right = sorted([shifts_mod[x] for x in range(1, piece_w, 2)],
                                  reverse=True)
            shifts = shifts_left + shifts_right
            for col_num, col_ind in enumerate(range(piece_w)):
                col = pic_piece[:, col_ind: col_ind + 1, :]
                col = np.roll(col, axis=0, shift=shifts[col_num])
                piece_rearranged.append(col)
            piece_shifted = np.concatenate(piece_rearranged, axis=1)
            processed.append(piece_shifted)
        # merge shifted elements back
        self.im_arr = np.concatenate(processed, axis=1)
        self.im_arr = tf.resize(self.im_arr, (w, h))

    def __apply_glitter(self, value):
        """Apply glitter."""
        dots = []  # fill this list with dot coordinates
        _dot_size = 3
        w, h, _ = self.im_arr.shape
        for _ in range(value):
            # just randomly select some coordinates
            dx = random.choice(range(_dot_size, w - _dot_size))
            dy = random.choice(range(_dot_size, h - _dot_size))
            dots.append((dx, dy))
        for dot in dots:
            self.im_arr[dot[0] - 1: dot[0], dot[1] - 3: dot[1] + 3, :] = 1

    def shift_hue(self, value):
        """Shift image hue in HSV."""
        img_hsv = color.rgb2hsv(self.im_arr)
        img_hsv[..., 0] += value
        img_hsv[..., 0] -= 1
        self.im_arr = color.hsv2rgb(img_hsv)

    def __apply_rb_shift(self, value, non_self_pic=None):
        """Draw chromatic aberrations."""
        if non_self_pic is None:
            _init_shape = self.im_arr.shape
            proc_pic = self.im_arr
        else:  # apply this filter to something else:
            self.v("Applying RGB shift to non-self.im_arr picture!")
            _init_shape = non_self_pic.shape
            proc_pic = non_self_pic
        # extract different channels
        red = proc_pic[:, :, 0]
        green = proc_pic[:, :, 1]
        blue = proc_pic[:, :, 2]

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
        if non_self_pic is None:
            self.im_arr = np.concatenate((red_n, green_n, blue_n), axis=2)
            # reshape it back
            self.im_arr = tf.resize(self.im_arr, (_init_shape[0], _init_shape[1]))
            self.v(f"Successfully applied {RB_SHIFT} filter")
            return None
        else:
            # return image (not self.im_arr)
            upd_img = np.concatenate((red_n, green_n, blue_n), axis=2)
            upd_img = tf.resize(upd_img, (_init_shape[0], _init_shape[1]))
            self.v("Applied RGB shift to non-self.im_arr image")
            return upd_img

    def __parse_mp3_attrs(self, attrs):
        """Parse mp3-related options."""
        self.v("Defining mp3-compression parameters")
        attrs_dict = {ADD_NOISE: False,
                      SOUND_QUALITY: 8,
                      BITRATE: 16,
                      INTENSIFY: False,
                      GLITCH_SOUND: False,
                      SHIFT_SIZE: int(self.im_arr.shape[1] / 2.35)
                      }
        # correct shift size if bitrate is pretty high
        if attrs_dict[BITRATE] >= 64:
            attrs_dict[SHIFT_SIZE] = 0
        avail_keys = set(attrs_dict.keys())
        # re-define default params
        for k, v in attrs.items():
            if k not in avail_keys:
                continue
            if v is None:
                continue
            self.v(f"Set param {k} to {v}")
            attrs_dict[k] = v
        # sanity checks
        if attrs_dict[SOUND_QUALITY] < 1 or attrs_dict[SOUND_QUALITY] > 10:
            self.__die(f"Sound quality must be in [1..10]")
        return attrs_dict

    def __add_noise(self):
        """Add noise to image (intensifies the effect)."""
        self.im_arr = util.random_noise(self.im_arr, mode="speckle")

    def mp3_compression(self, attrs):
        """Compress and decompress the image using mp3 algorithm.
        
        attrs -> a dictionary with additional parameters.
        """
        # split image in channels
        orig_image_ = self.im_arr.copy()
        self.v("Applying mp3 compression")
        mp3_attrs = self.__parse_mp3_attrs(attrs)
        self.__add_noise() if mp3_attrs[ADD_NOISE] else None
        # apply gamma correction upfront
        self.im_arr = exposure.adjust_gamma(image=self.im_arr, gain=self.gamma)
        w, h, _ = self.im_arr.shape
        # after the mp3 compression the picture shifts, need to compensate that
        # however, if bitrate >= 64 it doesn't actually happen
        red = self.im_arr[:, :, 0]
        green = self.im_arr[:, :, 1]
        blue = self.im_arr[:, :, 2]
        channels = (red, green, blue)
        glitched_channels = []
        # process them separately
        for num, channel in enumerate(channels, 1):
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
            mp3_compr_cmd = f'{self.lame_bin} -r --unsigned -s 16 -q {mp3_attrs[SOUND_QUALITY]} ' \
                            f'--resample 16 --bitwidth 8 -b {mp3_attrs[BITRATE]} ' \
                            f'-m m {raw_chan_} "{mp3_compr_}"'
            mp3_decompr_cmd = f'{self.lame_bin} --decode -x -t "{mp3_compr_}" {mp3_decompr_}'

            # call compress-decompress commands
            self.__call_proc(mp3_compr_cmd)
            # if required: change mp3 stream itself
            self.__glitch_sound(mp3_compr_, num, mp3_attrs) if mp3_attrs[GLITCH_SOUND] else None
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
            glitched_channels.append(glitched_channel)
        self.v("Concatenation of the mp3d image + rolling + adjust contrast")
        self.im_arr = np.concatenate(glitched_channels, axis=2)
        self.im_arr = np.roll(a=self.im_arr, axis=1, shift=mp3_attrs[SHIFT_SIZE])
        perc_left, perc_right = np.percentile(self.im_arr, (5, 95))
        self.im_arr = exposure.rescale_intensity(self.im_arr, in_range=(perc_left, perc_right))
        self.__remove_temp_files()  # don't need them anymore
        self.__intensify(orig_image_) if mp3_attrs[INTENSIFY] else None
    
    def __glitch_sound(self, mp3_path, ch_num, opts):
        """Change mp3 file directly."""
        self.v(f"Changing sound stream in {mp3_path} directly")
        if AudioSegment is None:
            self.__die("__glitch_sound requires Audiosegment (not imported)")
        x, y, _ = self.im_arr.shape
        # read sound file, get array of bytes (not a list!)
        sound = AudioSegment.from_mp3(mp3_path)
        last_ind = x * y  # sound array is a bit longer than image size
        entire_sound_array = np.array(sound.get_array_of_samples())
        # some processing here
        # final step: convert np array back to array
        new_array = array("h", entire_sound_array)
        new_sound = sound._spawn(new_array)
        new_sound.export(mp3_path, format='mp3')

    def __intensify(self, orig_image):
        """Intensify mp3 glitch using differences with original image."""
        self.v("Increasing mp3 glitch intensity")
        diff = self.im_arr - orig_image
        diff[diff < 0] = 0
        diff_hsv = color.rgb2hsv(diff)
        diff_hsv[..., 1] *= 5
        diff_hsv[..., 2] *= 2.5
        diff_hsv[diff_hsv >= 1.0] = 1.0
        diff = color.hsv2rgb(diff_hsv)
        self.im_arr += diff
        self.im_arr[self.im_arr > 1.0] = 1.0

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

    def save(self, path_):
        """Save the resulting image."""
        self.v(f"Saving image to: {path_}")
        io.imsave(fname=path_, arr=img_as_ubyte(self.im_arr))

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
        return [lst[x: x + n] for x in iter(range(0, len(lst), n))]


def parse_args():
    """Parse cmd args."""
    app = argparse.ArgumentParser()
    app.add_argument("input", help="Input image")
    app.add_argument("output", help="Output image")
    app.add_argument("--size", default=1000, type=int, help="Image size (long side)")
    app.add_argument("--verbose", "--v1", action="store_true", dest="verbose",
                     help="Verbosity mode on.")
    # filters
    app.add_argument(f"--{RB_SHIFT}", "-r", default=0, type=int,
                     help="RGB aberrations, the bigger value -> the higher intensity")
    app.add_argument(f"--{GLITTER}", "-g", default=0, type=int,
                     help="Add glitter, The bigger value -> the bigger sparks")
    app.add_argument(f"--{VERT_STREAKS}", "-v", action="store_true", dest=VERT_STREAKS,
                     help="Add vertical streaks")
    app.add_argument(f"--{ADD_TEXT}", "-t", default=None,
                     help="Add text (position is random)")
    app.add_argument("--text_position", "--tp", type=str,
                     help="Pre-define text coordinates (left corner) "
                          "two comma-separated values like 100,50")
    app.add_argument(f"--{ADD_RAINBOW}", "-a", dest=ADD_RAINBOW, action="store_true",
                     help="Add a rainbow!")
    # mp3-compression params
    app.add_argument(f"--compression_cycles", "--cc", default=1, type=int,
                     help="Number of mp3 compression-decompression cycles, default 1")
    app.add_argument("--save_each_cycle", "--sec", default=None,
                     help="Save an image of each compression cycle, specify "
                          "a directory if this is a case")
    app.add_argument(f"--{STRETCHING}", "--st", action="store_true", dest=STRETCHING,
                     help="Apply stretching filter")
    app.add_argument(f"--{ADD_NOISE}", "-n", action="store_true", dest=ADD_NOISE,
                     help="Add random noise to increase glitch effect")
    app.add_argument(f"--{SOUND_QUALITY}", "-q", type=int, default=8,
                     help="Gleitzsch sound quality")
    app.add_argument(f"--{BITRATE}", "-b", type=int, default=16,
                     help="MP3 bitrate")
    app.add_argument(f"--{INTENSIFY}", "-i", action="store_true", dest=INTENSIFY,
                     help="Get diff between mp3 glitched/not glitched image and "
                          "intensify glitched channel")
    app.add_argument(f"--{GLITCH_SOUND}", "-s", action="store_true", dest=GLITCH_SOUND,
                     help="Modify intermediate mp3 files")
    app.add_argument(f"--{SHIFT_SIZE}", "--sz", default=None, type=int,
                     help="Mp3 compression produces a horizontally shifted image "
                          "This parameter controls shift size, overriding "
                          "automatically assigned values")
    if len(sys.argv) < 3:
        app.print_help()
        sys.exit(0)
    args_ = app.parse_args()
    return args_


if __name__ == "__main__":
    args = parse_args()
    gleitzsch = Gleitzsch(args.input, args.size, args.verbose)
    gleitzsch.apply_filters(vars(args))  # as a dict: filter id -> value
    if args.compression_cycles > 1 and args.save_each_cycle:
        os.mkdir(args.save_each_cycle) if not os.path.isdir(args.save_each_cycle) else None
    for i in range(args.compression_cycles):
        if args.compression_cycles > 1:
            sys.stderr.write(f"Compression cycle num {i + 1}/{args.compression_cycles}\n")
        gleitzsch.mp3_compression(vars(args))
        if args.save_each_cycle:
            filename = f"{str(i).zfill(4)}.jpg"
            path = os.path.join(args.save_each_cycle, filename)
            gleitzsch.save(path)
    gleitzsch.save(args.output)
