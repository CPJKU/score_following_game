
from __future__ import print_function

import os
import sys
import yaml
import shutil
import numpy as np

from sheet_manager.data_model.piece import Piece
from sheet_manager.alignments import align_score_to_performance


SYSTEM_HEIGHT = 200


def coordinates_to_onsets(alignment, mdict, note_events):
    """
    Compute onset to coordinate mapping (cood_id -> onset_id)
    """

    coord_to_onset = dict()
    coords = []

    for i, (m_objid, e_idx) in enumerate(alignment):

        # get note mungo and midi note event
        m, e = mdict[m_objid], note_events[e_idx]

        # keep mapping from coordinate list id to
        # onset event (to work for different renderings)
        coord_to_onset[i] = e_idx

        # get note coordinates
        cy, cx = m.middle
        coords.append([cy, cx])

    # convert to array
    coords = np.asarray(coords, dtype=np.float32)

    return coords, coord_to_onset


def systems_to_rois(sys_mungos, window_top=10, window_bottom=10):
    """
    Convert systems to rois
    """

    page_rois = np.zeros((0, 4, 2))
    for sys_mungo in sys_mungos:
        t, l, b, r = sys_mungo.bounding_box

        cr = (t + b) // 2

        r_min = cr - window_top
        r_max = r_min + window_top + window_bottom
        c_min = l
        c_max = r

        topLeft = [r_min, c_min]
        topRight = [r_min, c_max]
        bottomLeft = [r_max, c_min]
        bottomRight = [r_max, c_max]
        system = np.asarray([topLeft, topRight, bottomRight, bottomLeft])
        system = system.reshape((1, 4, 2))
        page_rois = np.vstack((page_rois, system))

    return page_rois.astype(np.int)


def stack_images(images, mungos_per_page, mdict):
    """
    Re-stitch image
    """
    stacked_image = images[0]
    stacked_page_mungos = [m for m in mungos_per_page[0]]

    row_offset = stacked_image.shape[0]

    for i in range(1, len(images)):

        # append image
        stacked_image = np.concatenate((stacked_image, images[i]))

        # update coordinates
        page_mungos = mungos_per_page[i]
        for m in page_mungos:
            m.x += row_offset
            stacked_page_mungos.append(m)
            mdict[m.objid] = m

        # update row offset
        row_offset = stacked_image.shape[0]

    return stacked_image, stacked_page_mungos, mdict


def unwrap_sheet_image(image, system_mungos, mdict):
    """
    Unwrap all systems of sheet image to a single "long row"
    """

    # get rois from page systems
    rois = systems_to_rois(system_mungos, window_top=SYSTEM_HEIGHT // 2, window_bottom=SYSTEM_HEIGHT // 2)

    width = image.shape[1] * rois.shape[0]
    window = rois[0, 3, 0] - rois[0, 0, 0]

    un_wrapped_coords = dict()
    un_wrapped_image = np.zeros((window, width), dtype=np.uint8)

    # make single staff image
    x_offset = 0
    img_start = 0
    for j, sys_mungo in enumerate(system_mungos):

        # get current roi
        r = rois[j]

        # fix out of image errors
        pad_top = 0
        pad_bottom = 0
        if r[0, 0] < 0:
            pad_top = np.abs(r[0, 0])
            r[0, 0] = 0

        if r[3, 0] >= image.shape[0]:
            pad_bottom = r[3, 0] - image.shape[0]

        # get system image
        system_image = image[r[0, 0]:r[3, 0], r[0, 1]:r[1, 1]]

        # pad missing rows and fix coordinates
        system_image = np.pad(system_image, ((pad_top, pad_bottom), (0, 0)), mode='edge')

        img_end = img_start + system_image.shape[1]
        un_wrapped_image[:, img_start:img_end] = system_image

        # get noteheads of current staff
        staff_noteheads = [mdict[i] for i in sys_mungo.inlinks if mdict[i].clsname == 'notehead-full']

        # compute unwraped coordinates
        for n in staff_noteheads:
            n.x -= r[0, 0]
            n.y += x_offset - r[0, 1]
            un_wrapped_coords[n.objid] = n

        x_offset += (r[1, 1] - r[0, 1])
        img_start = img_end

    # get relevant part of unwrapped image
    un_wrapped_image = un_wrapped_image[:, :img_end]

    return un_wrapped_image, un_wrapped_coords


def prepare_piece_data(collection_dir, piece_name):
    """

    :param collection_dir:
    :param piece_name:
    :return:
    """

    # piece loading
    piece = Piece(root=collection_dir, name=piece_name)
    score = piece.load_score(piece.available_scores[0])

    # get mungos
    mungos = score.load_mungos()
    mdict = {m.objid: m for m in mungos}
    mungos_per_page = score.load_mungos(by_page=True)

    # load images
    images = score.load_images()

    # stack sheet images
    image, page_mungos, mdict = stack_images(images, mungos_per_page, mdict)

    # get only system mungos for unwrapping
    system_mungos = [c for c in page_mungos if c.clsname == 'staff']
    system_mungos = sorted(system_mungos, key=lambda m: m.top)

    # unwrap sheet images
    un_wrapped_image, un_wrapped_coords = unwrap_sheet_image(image, system_mungos, mdict)

    # load performances
    performance_key = piece.available_performances[0]

    # load current performance
    performance = piece.load_performance(performance_key)

    # running the alignment procedure
    alignment = align_score_to_performance(score, performance)

    # note events
    note_events = performance.load_note_events()

    # load spectrogram
    spec = performance.load_spectrogram()

    # compute onset to coodinate mapping
    coords, coord_to_onset = coordinates_to_onsets(alignment, un_wrapped_coords, note_events)

    return un_wrapped_image, coords, coord_to_onset


def load_audio_score_retrieval():
    """
    Load alignment data
    """

    # piece directory
    collection_dir = '/media/matthias/Data/msmd'

    # load split file
    split_file = "/home/matthias/cp/src/sheet_manager/sheet_manager/splits/all_split.yaml"
    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl)

    # process split
    for split_set in ["train", "valid", "test"]:

        for i_piece, piece_name in enumerate(split[split_set]):
            print("%d / %d" % (i_piece, len(split[split_set])), end="\r")
            sys.stdout.flush()

            piece_image, coords, piece_c2o_map = prepare_piece_data(collection_dir, piece_name)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(piece_image, cmap="gray")
            # plt.plot(coords[:, 1], coords[:, 0], 'co')
            # plt.colorbar()
            # plt.show()

            # dump rl data
            midi_src = os.path.join(collection_dir, piece_name, piece_name + '.midi')
            midi_dst = os.path.join('/home/matthias/mounts/home@rechenknecht5/shared/datasets/score_following_game/msmd_all/msmd_all_%s' % split_set, piece_name + '.mid')
            shutil.copy(midi_src, midi_dst)

            dump_file = midi_dst.replace('.mid', '.npz')
            np.savez(dump_file, sheet=piece_image, coords=coords, coord2onset=np.asarray([piece_c2o_map]))


if __name__ == "__main__":
    """ main """
    load_audio_score_retrieval()
