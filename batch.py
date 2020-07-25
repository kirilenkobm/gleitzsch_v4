#!/usr/bin/env python3
"""Batch processing."""
import sys
import os
import subprocess
from subprocess import DEVNULL

__author__ = "Bogdan Kirilenko, 2020"
__version__ = 4.0

GLEITZSCH = "./gleitzsch.py"


def main():
    """Sequentially glitch an entire directory."""
    argv = sys.argv
    if len(argv) < 3:
        sys.exit(f"Usage: {argv[0]} [input_dir] [output_dir] [--options]")
    if len(argv) == 3:
        input_dir = argv[1]
        output_dir = argv[2]
        opts = ""
    elif len(argv) > 3:
        input_dir = argv[1]
        output_dir = argv[2]
        opts = " ".join(argv[3:])
    
    input_files = os.listdir(input_dir)
    print(f"There are {len(input_files)} files in the batch")
    done, failed = 0, 0
    for filename in input_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        cmd = f"{GLEITZSCH} {input_path} {output_path} {opts}"
        print(f"Calling {cmd}")
        rc = subprocess.call(cmd, shell=True, stderr=DEVNULL, stdout=DEVNULL)
        if rc == 0:
            done += 1
        else:
            failed += 1
    print(f"Done; success: {done}; failed: {failed}")

if __name__ == "__main__":
    main()
