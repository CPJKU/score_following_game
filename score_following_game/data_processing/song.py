
import numpy as np
import os
import pretty_midi as pm
import score_following_game.data_processing.utils as utils

from scipy import interpolate
from score_following_game.data_processing.utils import fluidsynth, merge_onsets, midi_to_onsets, midi_reset_instrument,\
    midi_reset_start, pad_representation, wav_to_spec


class SongBase(object):
    def __init__(self, score, perf, song_name: str, config: dict, sound_font_default_path: str):

        self.config = config

        # parse config
        self.score_shape = config['score_shape']
        self.perf_shape = config['perf_shape']
        self.spec_representation = config['spec_representation']
        self.spectrogram_params = config['spectrogram_params']

        self.target_frame = config['target_frame']

        self.aggregate = config.get('aggregate', True)

        self.fps = self.spectrogram_params['fps']
        self.song_name = song_name

        self.spec_processing = utils.midi_to_spec_otf

        self.sound_font_default_path = sound_font_default_path

        self.sheet, self.coords, self.coords2onsets = score

        # pitch range
        self.pitch_range = range(config['pitch_range'][0], config['pitch_range'][1])

        # save the original MIDI
        self.org_perf_midi = perf

        # variable for the current performance
        self.cur_perf, self.perf_annotations = None, None

        # variable for the current score
        self.score = None

    def prepare_perf_representation(self, midi: pm.PrettyMIDI, padding: int, sound_font_path=None) -> dict:
        """Prepares a given midi for the later use inside of a data pool

        Parameters
        ----------
        midi : pm.PrettyMIDI
            the midi for which a representation should be created
        padding : int
            integer determining by how much the representation and onsets should be padded
        sound_font_path : str
            path to the sound font file

        Returns
        -------
        representation_dict : dict,
            'midi' -> pm.PrettyMidi, the midi file from the input
            'representation' -> np.ndarray, either a piano roll or spectrogram representation of the midi
            'representation_padded' -> the padded representation
            'onsets' -> np.ndarray, list of onsets
            'onsets_padded' -> the padded list of onsets
        """

        # add an empty second instrument to midi if not already available
        if len(midi.instruments) < 2:
            midi.instruments.append(pm.Instrument(1))

        onsets = midi_to_onsets(midi, self.fps, instrument_idx=None, unique=False)
        onsets, self.coords = merge_onsets(onsets, self.coords, self.coords2onsets)

        # extract spectrogram from synthesized audio
        # representation = midi_to_spec_otf(midi, self.spectrogram_params, sound_font_path=sound_font_path)
        representation = self.spec_processing(midi, self.spectrogram_params, sound_font_path=sound_font_path)

        # pad representation at the beginning and end
        representation_padded, onsets_padded = pad_representation(representation, onsets, padding)

        representation_dict = {'midi': midi, 'representation': representation,
                               'representation_padded': representation_padded,
                               'onsets': onsets, 'onsets_padded': onsets_padded,
                               'sound_font': sound_font_path}

        return representation_dict

    def get_representation_excerpts(self, perf_frame_idx_pad: int, est_score_position: int) -> (np.ndarray, np.ndarray):
        """Get performance and score excerpts depending on the performance
        frame index and the estimated score position

        Parameters
        ----------
        perf_frame_idx_pad : int
            padded index of the performance frame one wants to get an excerpt of
        est_score_position : int
            estimated position in the score one wants to get an excerpt of

        Returns
        -------
        perf_representation_excerpt : np.ndarray
            excerpt of the performance (either piano roll or spectrogram)
        score_representation_excerpt : np.ndarray
            excerpt of the score (currently only piano roll)
        """

        # get performance excerpt
        offset = self.score_shape[2] // 2
        if self.target_frame == 'right_most':
            perf_representation_excerpt = \
                self.cur_perf['representation_padded'][..., (perf_frame_idx_pad - self.perf_shape[2]):perf_frame_idx_pad]
        else:
            perf_representation_excerpt = \
                self.cur_perf['representation_padded'][..., (perf_frame_idx_pad - offset):(perf_frame_idx_pad + offset)]

        # choose staff excerpt range
        r0 = self.score['representation_padded'].shape[1]//2 - self.score_shape[1]//2
        r1 = r0 + self.score_shape[1]

        c0 = int(est_score_position - self.score_shape[2]//2)
        c1 = c0 + self.score_shape[2]
        # get score excerpt (centered around last onset)
        # score_representation_excerpt = self.score['representation_padded'][:, r0:r1,
        #                                int(est_score_position - offset + self.score_shape[2]):
        #                                int(est_score_position + offset + self.score_shape[2])]
        score_representation_excerpt = self.score['representation_padded'][:, r0:r1, c0:c1]

        return perf_representation_excerpt, score_representation_excerpt

    def get_perf_sf(self):
        return self.cur_perf['sound_font']

    def get_score_midi(self) -> pm.PrettyMIDI:
        return self.score['midi']

    def get_score_representation(self) -> np.ndarray:
        return self.score['representation']

    def get_score_representation_padded(self) -> np.ndarray:
        return self.score['representation_padded']

    def get_score_onsets(self) -> np.ndarray:
        return self.score['onsets']

    def get_perf_midi(self) -> pm.PrettyMIDI:
        return self.cur_perf['midi']

    def get_perf_audio(self, fs=44100):
        """Renders the current performance using FluidSynth and returns the waveform."""
        midi_synth = fluidsynth(self.cur_perf['midi'], fs=fs, sf2_path=self.cur_perf['sound_font'])

        # let it start at the first onset position
        midi_synth = midi_synth[int(self.cur_perf['onsets'][0] * fs / self.fps):]

        return midi_synth, fs

    def get_perf_representation(self) -> np.ndarray:
        return self.cur_perf['representation']

    def get_perf_representation_padded(self) -> np.ndarray:
        return self.cur_perf['representation_padded']

    def get_perf_onsets(self) -> np.ndarray:
        return self.cur_perf['onsets']

    def get_perf_onsets_padded(self) -> np.ndarray:
        return self.cur_perf['onsets_padded']

    def get_score_onsets_padded(self) -> np.ndarray:
        return self.score['onsets_padded']

    def get_num_of_perf_onsets(self) -> int:
        return len(self.cur_perf['onsets'])

    def get_num_of_score_onsets(self) -> int:
        return len(self.score['onsets'])

    def get_score_onset(self, idx) -> int:
        return self.score['onsets'][idx]

    def get_perf_onset(self, idx) -> int:
        return self.cur_perf['onsets'][idx]

    def get_true_score_position(self, perf_frame):
        """Use the mapping between performance and score to interpolate the score position,
           given the current performance position.

        Parameters
        ----------
        perf_frame : int
            Index of the performance frame.

        Returns
        -------
        true_score_position : float
            Interpolated position in the score.
            We keep a floating point value for the sake of distance calculations.
            However, in terms of frame idx, this does not make sense...
        """
        if perf_frame < self.cur_perf['onsets_padded'][0]:
            # No score onset yet...stay at first score position
            true_score_position = self.cur_perf['interpolation_fnc'](self.cur_perf['onsets_padded'][0])
        elif perf_frame > self.cur_perf['onsets_padded'][-1]:
            # Performance is over, stay with last onset
            true_score_position = self.cur_perf['interpolation_fnc'](self.cur_perf['onsets_padded'][-1])
        else:
            try:
                true_score_position = self.cur_perf['interpolation_fnc'](perf_frame)
            except ValueError:
                print(self.song_name, perf_frame)

        return true_score_position

    def get_song_name(self) -> str:
        return self.song_name

    def create_interpol_fnc(self, onsets_perf, onsets_score) -> interpolate.interp1d:
        """Mapping from performance positions to positions in the score."""

        interpol_fnc = None

        try:
            interpol_fnc = interpolate.interp1d(onsets_perf, onsets_score)
        except ValueError as e:
            print('There was a problem with song {}'.format(self.song_name))
            print(onsets_perf.shape, onsets_score.shape)
            print(onsets_perf)
            print(onsets_score)
            print(e)
        return interpol_fnc


class AudioSheetImgSong(SongBase):

    def __init__(self, score, perf: pm.PrettyMIDI, song_name: str, config: dict, sound_font_default_path: str):
        super(AudioSheetImgSong, self).__init__(score, perf, song_name, config, sound_font_default_path)

        self.cur_perf = self.prepare_perf_representation(self.org_perf_midi, padding=self.perf_shape[2],
                                                         sound_font_path=self.sound_font_default_path)

        self.sheet_padded, self.coords_padded = pad_representation(self.sheet, self.coords[:, 1], self.score_shape[2],
                                                                   pad_value=self.sheet.max())  # pad sheet white
        self.sheet = np.expand_dims(self.sheet, 0)
        self.sheet_padded = np.expand_dims(self.sheet_padded, 0)

        self.cur_perf['interpolation_fnc'] = self.create_interpol_fnc(self.cur_perf['onsets_padded'], self.coords_padded)

        self.score = {'representation': self.sheet, 'representation_padded': self.sheet_padded,
                      'onsets': self.cur_perf['onsets'], 'onsets_padded': self.cur_perf['onsets_padded'],
                      'coords': self.coords, 'coords_padded': self.coords_padded}


class RPWAudioSheetImgSong(SongBase):
    """
    Class representing
    """
    def __init__(self, score, perf, song_name: str, config: dict):
        super(RPWAudioSheetImgSong, self).__init__(score, perf, song_name, config, sound_font_default_path='')

        # prepare performance from recording
        self.path_perf = perf
        midi_perf = midi_reset_instrument(pm.PrettyMIDI(self.path_perf.replace('.wav', '.mid')), id=0)
        midi_perf = midi_reset_start(midi_perf)

        self.cur_perf = self.prepare_perf_representation(perf, padding=self.perf_shape[2], perf_midi=midi_perf)

        self.sheet_padded, self.coords_padded = pad_representation(self.sheet, self.coords[:, 1], self.score_shape[2],
                                                                   pad_value=self.sheet.max())  # pad sheet white
        self.sheet = np.expand_dims(self.sheet, 0)
        self.sheet_padded = np.expand_dims(self.sheet_padded, 0)

        self.cur_perf['interpolation_fnc'] = self.create_interpol_fnc(self.cur_perf['onsets_padded'], self.coords_padded)

        self.score = {'representation': self.sheet, 'representation_padded': self.sheet_padded,
                      'onsets': self.cur_perf['onsets'], 'onsets_padded': self.cur_perf['onsets_padded'],
                      'coords': self.coords, 'coords_padded': self.coords_padded}

    def prepare_perf_representation(self, path_audio: str, padding: int, perf_midi: pm.PrettyMIDI) -> dict:
        """Prepares a given audio recording for the use in the datapool

        Parameters
        ----------
        path_audio : str
            Path to the audio recording.
        padding : int
            integer determining by how much the representation and onsets should be padded
        perf_midi : PrettyMIDI
            performance MIDI file

        Returns
        -------
        representation_dict : dict,
            'representation' -> np.ndarray, either a piano roll or spectrogram representation of the midi
            'representation_padded' -> the padded representation
            'onsets' -> np.ndarray, list of onsets
            'onsets_padded' -> the padded list of onsets
        """

        # load performance onsets from annotations
        onsets = midi_to_onsets(perf_midi, self.fps, instrument_idx=None, unique=False)
        onsets, self.coords = merge_onsets(onsets, self.coords, self.coords2onsets)

        # extract spectrogram from synthesized audio
        representation = wav_to_spec(path_audio, self.spectrogram_params)

        # pad representation at the beginning and end
        representation_padded, onsets_padded = pad_representation(representation, onsets, padding)

        representation_dict = {'representation': representation,
                               'representation_padded': representation_padded,
                               'onsets': onsets,
                               'onsets_padded': onsets_padded}

        return representation_dict

    def get_perf_audio(self, fs=44100):
        """use existing waveform."""
        from scipy.io import wavfile
        fs, data = wavfile.read(self.path_perf)
        return data, fs


def load_song(config: dict, cur_path_score, cur_path_perf, real_perf=False) -> SongBase:

    cur_song_name = os.path.splitext(os.path.basename(os.path.normpath(cur_path_score)))[0]

    npzfile = np.load(cur_path_score, allow_pickle=True)
    score = (npzfile['sheet'], npzfile['coords'], npzfile['coord2onset'][0])

    sound_font_default_path = config['default_sf']

    if real_perf == 'wav':

        cur_path_perf = cur_path_perf.replace('.mid', '.wav')
        return RPWAudioSheetImgSong(score, cur_path_perf, cur_song_name, config)

    else:

        cur_midi_perf = midi_reset_instrument(pm.PrettyMIDI(cur_path_perf), id=0)
        return AudioSheetImgSong(score, cur_midi_perf, cur_song_name, config,
                                 sound_font_default_path=sound_font_default_path)


def create_song(config, song_name, score, perf, default_sf_path) -> SongBase:

    perf = midi_reset_instrument(perf, id=0) if isinstance(perf, pm.PrettyMIDI) else perf
    score = midi_reset_instrument(score, id=0) if isinstance(score, pm.PrettyMIDI) else score

    return AudioSheetImgSong(score, perf, song_name, config, sound_font_default_path=default_sf_path)
