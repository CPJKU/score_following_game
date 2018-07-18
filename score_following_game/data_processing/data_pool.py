from __future__ import print_function

import yaml
import os
import numpy as np

from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    LogarithmicFilterbank
from madmom.processors import SequentialProcessor

import madmom.utils.midi as mm_midi
from scipy import interpolate
from scipy.sparse import csr_matrix

from score_following_game.config.settings import SOUND_FONT_PATH
from score_following_game.data_processing.compression import CompressedArrayList


if not os.path.exists(SOUND_FONT_PATH):
    """
    This extracts the zip-compressed soundfont used in the game.
    (Required due to a file size limitation of github.)
    """
    import zipfile
    print("Extracting soundfont file %s ..." % SOUND_FONT_PATH)
    zip_ref = zipfile.ZipFile(SOUND_FONT_PATH + ".zip", 'r')
    zip_ref.extractall(os.path.dirname(SOUND_FONT_PATH))
    zip_ref.close()


class RLScoreFollowPool(object):
    """
    Data Pool for spectrogram to sheet snippet hashing
    """

    def __init__(self, sheets, coords, coord2onsets, midi_files_path, config, prioritized_sampling=False):
        """
        Constructor
        """
        self.sheets = sheets
        self.coords = coords
        self.coord2onsets = coord2onsets
        self.midis = load_midis(midi_files_path)
        self.spectrograms, self.audio_paths = midi_files_to_spectrograms(midi_files_path, config["spectrogram_params"])
        self.onsets = midi_files_to_onsets(self.midis, config["spectrogram_params"]["fps"])
        self.spec_context = config["spec_context"]
        self.sheet_context = config["sheet_context"]
        self.spectrogram_params = config["spectrogram_params"]
        self.target_frame = config["target_frame"]
        self.staff_height = config["staff_height"]

        self.frequency_bins = self.spectrograms[0].shape[0]

        # pad spectrograms
        self.spec_pad = 2 * self.spec_context
        self.spectrograms, self.onsets = pad_spectrograms(self.spectrograms, self.onsets, self.spec_pad)

        # pad sheets
        self.sheet_pad = 2 * self.sheet_context
        self.sheets, self.coords = pad_sheets(self.sheets, self.coords, self.sheet_pad)

        # compress sheets
        # self.sheets = CompressedArrayList(self.sheets)
        # for i in range(len(self.sheets)):
        #     self.sheets[i] = csr_matrix(255 - self.sheets[i])

        # merge onsets taking place at the same time
        # required for interpolation
        for i in range(len(self.onsets)):
            self.onsets[i], self.coords[i] = \
                merge_onsets(self.onsets[i], self.coords[i], self.coord2onsets[i])

        # interpolate onsets / coordinates on spectrogram frame level
        self.interpolationFunctions = create_interpol_fncs(self.coords, self.onsets)

        self.spec_offset = self.spec_context // 2
        self.sheet_offset = self.sheet_context // 2

        # initialize data generator
        self.sheet_id = None
        self.first_onset = None
        self.last_onset = None
        self.next_onset_idx = 0
        self.current_onset = None
        self.next_onset = None
        self.curr_spec_frame = None
        self.prioritized_sampling = prioritized_sampling
        self.tracking_ratios = np.zeros(len(sheets))

        self.sheet_coord = None
        self.true_sheet_coord = None

        # initialize controller
        self.sheet_speed = None

        # reset state generator
        self.reset()

    def reset(self):
        """ reset generator """

        # randomly select sheet
        if self.prioritized_sampling:
            eps = 0.05  # each piece should keep a small probability of being selected
            selection_prob = 1.0 - np.clip(self.tracking_ratios, 0.0, 1.0 - eps)
            selection_prob /= selection_prob.sum()
            self.sheet_id = np.random.choice(len(self.sheets), p=selection_prob)
        else:
            self.sheet_id = np.random.randint(0, len(self.sheets))

        self.first_onset = int(self.onsets[self.sheet_id][0])
        self.last_onset = int(self.onsets[self.sheet_id][-1])
        self.next_onset_idx = 0
        self.next_onset = self.first_onset

        # initialize sheet coordinates
        self.sheet_coord = self.coords[self.sheet_id][0, 1]
        self.sheet_coord = self.clip_coord(self.sheet_coord, self.sheets[self.sheet_id])
        self.true_sheet_coord = self.coords[self.sheet_id][0, 1]

        # reset sheet speed
        self.sheet_speed = 0

    def step(self, step_frame):
        """
        Perform time step

        @step_frame (int): current frame in spectrogram
        """

        # target onset centered rightmost frame
        self.curr_spec_frame = step_frame + self.spec_pad

        # update sheet coordinate
        self.sheet_coord += self.sheet_speed

        # clip coordinate to be within sheet bounds
        self.sheet_coord = self.clip_coord(self.sheet_coord, self.sheets[self.sheet_id])

        if self.curr_spec_frame == self.next_onset:

            self.next_onset_idx += 1
            self.current_onset = self.next_onset
            if self.next_onset_idx < len(self.onsets[self.sheet_id]):
                self.next_onset = self.onsets[self.sheet_id][self.next_onset_idx]

        # update tracking ratio
        self.tracking_ratios[self.sheet_id] = float(self.curr_spec_frame) / self.last_onset

    def observe(self):
        """ get next observation """

        # get spectrogram excerpt
        curr_spec = self.spectrograms[self.sheet_id]

        # target onset centered rightmost frame
        if self.target_frame == "right_most":
            spec_excerpt = curr_spec[:, self.curr_spec_frame - self.spec_context:self.curr_spec_frame]
        # target onset centered in spectrogram
        else:
            spec_excerpt = curr_spec[:, self.curr_spec_frame - self.spec_offset:self.curr_spec_frame + self.spec_offset]

        # get true image center
        if self.curr_spec_frame < self.onsets[self.sheet_id][0]:
            self.true_sheet_coord = self.interpolationFunctions[self.sheet_id](self.onsets[self.sheet_id][0])
        elif self.curr_spec_frame > self.onsets[self.sheet_id][-1]:
            self.true_sheet_coord = self.interpolationFunctions[self.sheet_id](self.onsets[self.sheet_id][-1])
        else:
            self.true_sheet_coord = self.interpolationFunctions[self.sheet_id](self.curr_spec_frame)

        # get sheet snippet
        # curr_sheet = 255 - self.sheets[self.sheet_id].todense()
        curr_sheet = self.sheets[self.sheet_id]

        c0 = int(self.sheet_coord - self.sheet_offset)
        c1 = c0 + self.sheet_context
        r0 = curr_sheet.shape[0] // 2 - self.staff_height // 2
        r1 = r0 + self.staff_height
        sheet_snippet = curr_sheet[r0:r1, c0:c1]

        return spec_excerpt, sheet_snippet

    def update_sheet_speed(self, speed_update):
        """ update sheet speed """
        self.sheet_speed += speed_update

    def tracking_error(self):
        """ compute distance between true and tracked sheet position """
        return self.sheet_coord - self.true_sheet_coord

    def get_true_sheet_coord(self):
        return self.true_sheet_coord

    def last_onset_reached(self):
        return self.curr_spec_frame == self.last_onset

    def get_current_song_timesteps(self):
        return self.spectrograms[self.sheet_id].shape[1]

    def get_current_song_name(self):
        return os.path.basename(self.audio_paths[self.sheet_id])[0:-4]

    def get_current_audio_file(self):
        return self.audio_paths[self.sheet_id]

    def get_current_song_onsets(self):
        return self.onsets[self.sheet_id]

    def in_onset_range(self, onset_window=0):

        # setup a window around the current and next onset because the next onset
        window_range = np.arange(-onset_window, onset_window+1)

        if self.current_onset is not None:
            window_curr = np.ones(len(window_range)) * self.current_onset + window_range
        else:
            window_curr = []

        window_next = np.ones(len(window_range))*self.next_onset+window_range

        # check if the current onset is within this window
        return self.curr_spec_frame in window_curr or self.curr_spec_frame in window_next

    def clip_coord(self, coord, sheet):
        """
        Clip coordinate to be within sheet bounds
        """

        coord = np.max([coord, self.sheet_offset])
        coord = np.min([coord, sheet.shape[1] - self.sheet_offset - 1])

        return coord


def merge_onsets(cur_onsets, stk_note_coords, coords2onsets):
    """ merge onsets occurring in the same frame """

    # get coordinate keys
    coord_ids = coords2onsets.keys()

    # init list of unique onsets and coordinates
    onsets, coords = [], []

    # iterate coordinates
    for i in coord_ids:

        # check if onset already exists in list
        if cur_onsets[coords2onsets[i]] not in onsets:
            coords.append(stk_note_coords[i])
            onsets.append(cur_onsets[coords2onsets[i]])

    # convert to arrays
    coords = np.asarray(coords, dtype=np.float32)
    onsets = np.asarray(onsets, dtype=np.float32)

    # # get unique onsets
    # cur_onsets, counts = np.unique(cur_onsets, return_counts=True)
    #
    # # iterate onsets and delete "duplicates"
    # for i in range(len(cur_onsets)):
    #     to_delete = slice(i + 1, i + counts[i])
    #     stk_note_coords = np.delete(stk_note_coords, to_delete, axis=0)
    #
    # # check if there is the same number of coordinates as onsets
    # assert len(cur_onsets) == len(stk_note_coords), "number of notes changed after onset merging"

    return onsets, coords


def pad_spectrograms(spectrograms, onsets, spec_pad):
    for i in range(len(spectrograms)):
        spectrograms[i] = np.pad(spectrograms[i], ((0, 0), (spec_pad, spec_pad)), mode="constant")
        onsets[i] = onsets[i] + spec_pad
    return spectrograms, onsets


def pad_sheets(sheets, coordinates, sheet_pad):
    for i in range(len(sheets)):
        sheets[i] = np.pad(sheets[i], ((0, 0), (sheet_pad, sheet_pad)), mode="constant",
                           constant_values=sheets[i].max())
        coordinates[i][:, 1] = coordinates[i][:, 1] + sheet_pad
    return sheets, coordinates


def midi_files_to_spectrograms(midi_files, spec_params):

    def render_audio(midi_file_path, sound_font):
        """
        Render midi to audio
        """

        # split file name and extention
        name, extention = midi_file_path.rsplit(".", 1)

        # set file names
        audio_file = name + ".wav"

        # synthesize midi file to audio
        cmd = "fluidsynth -F %s -O s16 -T wav %s %s" % (audio_file, sound_font, midi_file_path)
        os.system(cmd)

    def spec_from_midi(midi_file):

        sig_proc = SignalProcessor(num_channels=1, sample_rate=spec_params["sample_rate"])
        fsig_proc = FramedSignalProcessor(frame_size=spec_params["frame_size"], fps=spec_params["fps"])
        spec_proc = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=60, fmax=6000,
                                                 norm_filters=True, unique_filters=False)
        log_proc = LogarithmicSpectrogramProcessor()
        processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_proc])

        # print(midi_file)
        if not os.path.isfile(midi_file.replace('.mid', '.wav')):
            # render audio file from midi
            render_audio(midi_file, sound_font=SOUND_FONT_PATH)

        # compute spectrogram
        audio_path = midi_file.replace('.mid', '.wav')

        # if the spectrogram doesn't exist it will be computed and stored
        if not os.path.isfile(midi_file.replace('.mid', '.spec.npy')):
            spec = processor.process(audio_path).T
            np.save(midi_file.replace('.mid', '.spec'), spec)
        else:
            spec = np.load(midi_file.replace('.mid', '.spec.npy'))

        return spec

    spectrograms = []
    audio_paths = []

    for midi in midi_files:
        spectrograms.append(spec_from_midi(midi))
        audio_paths.append(midi.replace('.mid', '.wav'))

    return spectrograms, audio_paths


def load_midis(midi_files_path):

    midis = []
    for path in midi_files_path:
        midis.append(mm_midi.MIDIFile.from_file(path))

    return midis


def midi_files_to_onsets(midi_files, fps):

    def onsets_from_midi(midi_file):

        onsets = []
        for n in midi_file.notes():
            onset = int(np.ceil(n[0] * fps))
            onsets.append(onset)

        return np.sort(np.asarray(onsets)).astype(np.float32)

    onsets_list = []

    for midi in midi_files:
        onsets_list.append(onsets_from_midi(midi))

    return onsets_list


def create_interpol_fncs(coords, onsets):
    interpol_fncs = []

    for i in range(len(coords)):
        interpol_fncs.append(interpolate.interp1d(onsets[i], coords[i][:, 1]))

    return interpol_fncs


def load_game_config(config_file):
    with open(config_file, "rb") as fp:
        config = yaml.load(fp)
    return config


def get_data_pool(config, directory='test_sample', song_name=None):

    sheets = []
    coords = []
    coord2onsets = []
    midi_files = []

    for entry in search_directory(directory):

        if song_name is not None and song_name not in entry[0]:
            continue

        npzfile = np.load(entry[1])
        sheet = npzfile["sheet"]
        coordinates = npzfile["coords"]
        coord2onset = npzfile['coord2onset'][0]

        sheets.append(sheet)
        coords.append(coordinates)
        coord2onsets.append(coord2onset)
        midi_files.append(entry[0])

    rl_pool = RLScoreFollowPool(sheets, coords, coord2onsets, midi_files, config)

    return rl_pool


def get_data_pools(config, directory='test_sample'):

    pools = []
    for entry in search_directory(directory):
        npzfile = np.load(entry[1])
        sheet = npzfile["sheet"]
        coordinates = npzfile["coords"]
        coord2onset = npzfile['coord2onset'][0]

        pools.append(RLScoreFollowPool([sheet], [coordinates], [coord2onset], [entry[0]], config))

    return pools


def search_directory(directory):
    import os

    paths = []

    for dirname, dirnames, filenames in os.walk(directory):

        for filename in filenames:

            name = os.path.join(dirname, filename)

            if name.endswith('.mid'):
                name = name[0:name.find('.mid')]

                midi = name + '.mid'
                npz = name + '.npz'

                paths.append([midi, npz])

    return paths


if __name__ == "__main__":
    """ main """
    config = load_game_config('../game_configs/nottingham.yaml')
    pool = get_data_pool(config, directory='../test_sample')
