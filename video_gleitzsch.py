#!/usr/bin/env python3
"""Apply Gleitzsch to video."""
import argparse
import sys
import os
import cv2
import numpy as np
import imageio
from gleitzsch import Gleitzsch

RB_SHIFT = "rb_shift"
GLITTER = "glitter"
GAMMA_CORRECTION = "gammma_correction"
ADD_TEXT = "add_text"
VERT_STREAKS = "vert_streaks"

ADD_NOISE = "add_noise"
SOUND_QUALITY = "sound_quality"
BITRATE = "bitrate"
INTENSIFY = "intensify"
GLITCH_SOUND = "glitch_sound"
ADD_RAINBOW = "add_rainbow"
SHIFT_SIZE = "shift_size"


def parse_args():
    """Parse args."""
    app = argparse.ArgumentParser()
    app.add_argument("input", help="Input video file")
    app.add_argument("output", help="Video output")
    app.add_argument("--cut_frames", default=0, type=int,
                     help="Process only N first frames")
    app.add_argument("--shift", default=0, type=int,
                     help="Assign frame shift manually")
    app.add_argument("--size", default=640, type=int,
                     help="Image size")
    if len(sys.argv) < 3:
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    return args


def load_video(in_path):
    """Load video.

    Recipe taken from:
    https://stackoverflow.com/questions/42163058/how-to-turn-a-video-into-numpy-array
    """
    cap = cv2.VideoCapture(in_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frame_count  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf


def main():
    """Entry point."""
    args = parse_args()
    video = load_video(args.input)
    frames_num, x, y, _ = video.shape
    print(f"There are {frames_num} frames in the video {args.input}")
    print(f"Frame size: {x}x{y}")
    shift_size = args.shift if args.shift > 0 else int(args.size / 2.35)
    mp3_attrs_dict = {ADD_NOISE: False,
                      SOUND_QUALITY: 5,
                      BITRATE: 12,
                      INTENSIFY: False,
                      GLITCH_SOUND: False,
                      SHIFT_SIZE: shift_size
                      }

    filt_dict = {RB_SHIFT: 8,
                 GLITTER: None,
                 GAMMA_CORRECTION: None,
                 ADD_TEXT: None,
                 VERT_STREAKS: None,
                 ADD_RAINBOW: False,
                 }
    gif_frames = []

    for num, frame in enumerate(video, 1):
        print(f"Processing frame {num} / {frames_num}")
        frame_glitch = Gleitzsch(frame, size=args.size)
        frame_glitch.apply_filters(filt_dict)
        frame_glitch.mp3_compression(mp3_attrs_dict)
        frame_glitch.shift_hue(0.5)
        gif_frames.append(frame_glitch.im_arr)
        if args.cut_frames > 0 and num >= args.cut_frames:
            break
    
    imageio.mimsave(args.output, gif_frames)

if __name__ == "__main__":
    main()
