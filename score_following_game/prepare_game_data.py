
from __future__ import print_function

import os
import sys
import tarfile

FILE_PATH = "http://drive.jku.at/ssf/s/readFile/share/5691/-1682982240985647822/publicLink/score_following_game_data.tar.gz"


def download_file(source, destination):
    """
    Load file from url
    """
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    print("Downloading game data to %s ... " % destination, end="")
    sys.stdout.flush()
    urlretrieve(source, destination)
    print("done!")


def extract_file(tar_file):
    print("Extracting score images and MIDIs ... ", end="")
    sys.stdout.flush()
    tar = tarfile.open(tar_file)
    dst_dir = os.path.dirname(tar_file)
    tar.extractall(dst_dir)
    tar.close()
    print("done!")


if __name__ == '__main__':
    """ main """

    # add argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Prepare score following game data.')
    parser.add_argument('--destination_dir', help='path to data destination directory.', type=str, default=".")
    args = parser.parse_args()

    # create destination directory
    if not os.path.exists(args.destination_dir):
        os.makedirs(args.destination_dir)

    # get file name
    file_name = os.path.basename(FILE_PATH)

    # set destination path
    dst = os.path.join(args.destination_dir, file_name)
    download_file(FILE_PATH, dst)
    extract_file(dst)
