import glob
import itertools
import os
import tqdm

import numpy as np

from collections import Counter
from multiprocessing import get_context
from multiprocessing.managers import BaseManager
from scipy import interpolate
from score_following_game.data_processing.data_utils import wav_to_spec
from typing import List


class AudioSheetImgSong:
    def __init__(self, score, coords, onsets, perf, song_name: str, config: dict):

        # parse config
        self.score_excerpt_shape = tuple(config['score_shape'])
        self.perf_excerpt_shape = tuple(config['perf_shape'])
        self.spectrogram_params = config['spectrogram_params']

        self.target_frame = config['target_frame']

        self.song_name = song_name

        # prepare performance from recording
        self.perf_path = perf

        # start performance at first onset
        min_onset = np.min(onsets)
        onsets -= min_onset
        spec = wav_to_spec(self.perf_path, self.spectrogram_params)[..., min_onset:]

        # pad performance and onset
        self.performance = np.pad(spec, ((0, 0), (0, 0), (self.perf_excerpt_shape[2], 0)), mode='constant',
                                  constant_values=0)
        self.perf_onsets = onsets + self.perf_excerpt_shape[2]

        # pad score white
        self.score = np.pad(score, ((0, 0), (self.score_excerpt_shape[2], self.score_excerpt_shape[2])),
                            mode='constant', constant_values=score.max())
        self.coords = coords[:, 1] + self.score_excerpt_shape[2]

        self.score = np.expand_dims(self.score, 0)

        self.interpolation_fnc = interpolate.interp1d(self.perf_onsets, self.coords, bounds_error=False,
                                                      fill_value=(self.coords[0], self.coords[-1]))
        self.inverse_interpolation_fnc = interpolate.interp1d(self.coords, self.perf_onsets,
                                                              # bounds_error=False, kind='nearest',
                                                              bounds_error=False, kind='previous',
                                                              fill_value=(self.perf_onsets[0], self.perf_onsets[-1]))

    def get_perf_audio(self):
        """use existing waveform."""
        from scipy.io import wavfile
        fs, data = wavfile.read(self.perf_path)
        return data, fs

    def get_true_score_position(self, perf_frame):
        """
         Use the mapping between performance and score to interpolate the score position,
         given the current performance position.
        """
        return self.interpolation_fnc(perf_frame)

    def get_excerpts(self, perf_frame_idx_pad: int, est_score_position: int) -> (np.ndarray, np.ndarray):
        """
        Get performance and score excerpts depending on the performance
        frame index and the estimated score position

        Parameters
        ----------
        perf_frame_idx_pad : int
            padded index of the performance frame one wants to get an excerpt of
        est_score_position : int
            estimated position in the score one wants to get an excerpt of

        Returns
        -------
        perf_excerpt : np.ndarray
        score_excerpt : np.ndarray
        """

        # get performance excerpt
        offset = self.score_excerpt_shape[2] // 2
        if self.target_frame == 'right_most':
            perf_excerpt = \
                self.performance[..., (perf_frame_idx_pad - self.perf_excerpt_shape[2]):perf_frame_idx_pad]
        else:
            perf_excerpt = \
                self.performance[..., (perf_frame_idx_pad - offset):(perf_frame_idx_pad + offset)]

        # choose staff excerpt range
        r0 = self.score.shape[1] // 2 - self.score_excerpt_shape[1] // 2
        r1 = r0 + self.score_excerpt_shape[1]

        c0 = int(est_score_position - self.score_excerpt_shape[2] // 2)
        c1 = c0 + self.score_excerpt_shape[2]

        # get score excerpt (centered around last onset)
        score_excerpt = self.score[:, r0:r1, c0:c1]

        # check if the excerpts have the desired shape
        try:
            assert self.perf_excerpt_shape == perf_excerpt.shape and self.score_excerpt_shape == score_excerpt.shape
        except AssertionError as e:
            print('Encountered a shape mismatch.')
            print('Song name: {}, perf_frame_idx_pad: {}, est_score_position: {}'.format(self.song_name,
                                                                                         perf_frame_idx_pad,
                                                                                         est_score_position))
            print('Performance: desired shape = {}, actual shape= {}'.format(self.perf_excerpt_shape,
                                                                             perf_excerpt.shape))
            print('Score: desired shape = {}, actual shape= {}'.format(self.score_excerpt_shape,
                                                                       score_excerpt.shape))
            raise e

        return perf_excerpt, score_excerpt

    @property
    def num_of_frames(self):
        return self.performance.shape[-1]

    @property
    def last_onset(self):
        return int(self.perf_onsets[-1])

    @property
    def first_onset(self):
        return int(self.perf_onsets[0])


class SongPool:

    def __init__(self, songs):
        self.songs = songs
        self.length = len(songs)

    def get_length(self):
        return self.length

    def get_song(self, item):
        return self.songs[item]


def create_shared_songs_pool(songs):
    BaseManager.register('SongPool', SongPool)
    manager = BaseManager()
    manager.start()

    return manager.SongPool(songs)


def get_single_song_pool(params) -> List[SongPool]:

    config = params['config']
    song_name = params['song_name']
    directory = params.get('directory', 'test_sample')
    split = params['split']

    cur_path_score = os.path.join(directory, song_name + ".npz")

    songs = load_song({'score_path': cur_path_score,  'config': config, 'split': split})

    pools = []
    for song in songs:
        pools.append(SongPool([song]))

    return pools


def get_data_pools(config: dict, split: bool = False, directory: str = 'test_sample') -> List[SongPool]:
    """Get a list of data pools with each data pool containing only a single song from the directory

    Parameters
    ----------
    config : dict
        dictionary specifying the config for the data pool and songs
    split : bool
        flag indicating whether each page of a piece should be considered separately
    directory : str
        path to the directory containing the data that should be loaded

    Returns
    -------
    pools : List[RLScoreFollowPool]
        list of data pools
    """

    print('Load data pools...')

    # score_paths = list(glob.glob(os.path.join(directory, score_folder, '*.npz')))
    score_paths = list(glob.glob(os.path.join(directory, '*.npz')))

    params = [
        dict(
            song_name=os.path.splitext(os.path.basename(os.path.normpath(score_path)))[0],
            config=config,
            split=split,
            directory=directory,
        )
        for score_path in score_paths
    ]

    with get_context("spawn").Pool(8) as pool:
        data_pools = list(tqdm.tqdm(pool.imap(get_single_song_pool, params), total=len(params)))
    data_pools = list(itertools.chain(*data_pools))
    return data_pools


def load_song(params):

    config = params['config']
    score_path = params['score_path']
    perf_path = score_path.replace("npz", "wav")
    split = params['split']

    cur_song_name = os.path.splitext(os.path.basename(os.path.normpath(score_path)))[0]

    npzfile = np.load(score_path, allow_pickle=True)
    scores = npzfile["sheets"]
    coords, systems = list(npzfile["coords"]), list(npzfile['systems'])

    songs = []
    full_unrolled_score = []
    full_unrolled_coords = []
    full_onsets = []

    for page in range(len(scores)):

        score = scores[page]
        page_coords = list(filter(lambda x: x['page_nr'] == page, coords))

        if len(page_coords) > 0:
            unrolled_score, unrolled_coords, onsets = unroll_score(score, page_coords, systems, page)

            if split:
                song = AudioSheetImgSong(unrolled_score, unrolled_coords, onsets, perf_path,
                                         cur_song_name+f"_page_{page}", config)
                songs.append(song)
            else:
                full_unrolled_score.append(unrolled_score)
                full_unrolled_coords.append(unrolled_coords)
                full_onsets.append(onsets)

    if not split:

        unrolled_score = np.hstack(full_unrolled_score)
        onsets = np.concatenate(full_onsets)

        for page in range(1, len(full_unrolled_score)):
            full_unrolled_coords[page][:, 1] += full_unrolled_score[page - 1].shape[-1]

        unrolled_coords = np.concatenate(full_unrolled_coords)

        song = AudioSheetImgSong(unrolled_score, unrolled_coords, onsets,
                                 perf_path, cur_song_name, config)
        songs.append(song)

    return songs


def unroll_score(page, coords, systems, page_nr):

    unrolled_score = []

    unrolled_coords = []
    width_offset = 0
    for system_idx, system in enumerate(systems):

        if system['page_nr'] != page_nr:
            continue
        system_coords = list(filter(lambda x: x['system_idx'] == system_idx, coords))

        x_from = max(int(system['x'] - system['w']//2), 0)
        x_to = min(int(system['x'] + system['w']//2), page.shape[1])

        y_from = max(int(system['y'] - 100), 0)
        y_to = min(int(system['y'] + 100), page.shape[0])

        for c in system_coords:
            c['note_x'] += width_offset - x_from
            unrolled_coords.append(c)

        excerpt = page[y_from:y_to, x_from:x_to]
        if excerpt.shape[0] != 200:
            excerpt = np.pad(excerpt, (200-excerpt.shape[0], 0), constant_values=255)

        unrolled_score.append(excerpt)

        width_offset += unrolled_score[-1].shape[-1]

    unrolled_score = np.hstack(unrolled_score)

    onsets = []
    for i in range(len(unrolled_coords)):
        # onset time to frame
        unrolled_coords[i]['onset'] = int(unrolled_coords[i]['onset'] * 20)
        onsets.append(unrolled_coords[i]['onset'])

    onsets = np.asarray(onsets, dtype=np.int)

    onsets = np.unique(onsets)
    coords_new = []
    for onset in onsets:
        onset_coords = list(filter(lambda x: x['onset'] == onset, unrolled_coords))

        onset_coords_merged = {}
        for entry in onset_coords:
            for key in entry:
                if key not in onset_coords_merged:
                    onset_coords_merged[key] = []
                onset_coords_merged[key].append(entry[key])

        # get system, bar and page with most notes in it
        system_idx = int(Counter(onset_coords_merged['system_idx']).most_common(1)[0][0])
        note_x = np.mean(
            np.asarray(onset_coords_merged['note_x'])[np.asarray(onset_coords_merged['system_idx']) == system_idx])
        page_nr = int(Counter(onset_coords_merged['page_nr']).most_common(1)[0][0])

        # set y to staff center
        note_y = -1.0
        if note_x > 0:
            note_y = systems[system_idx]['y']
        coords_new.append([note_y, note_x, page_nr])

    coords_new = np.asarray(coords_new)

    return unrolled_score, coords_new, onsets


def load_songs(config, directory, split=False):

    params = []

    for score_path in glob.glob(os.path.join(directory, '*.npz')):
        params.append({'score_path': score_path, 'config': config, 'split': split})

    with get_context("fork").Pool(8) as pool:
        songs = list(tqdm.tqdm(pool.imap(load_song, params), total=len(params)))

    songs = list(itertools.chain(*songs))
    return songs
