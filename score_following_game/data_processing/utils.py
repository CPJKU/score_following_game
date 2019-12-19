import glob
import librosa
import os
import tempfile
import time
import tqdm
import yaml

import numpy as np
import pretty_midi as pm

from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    LogarithmicFilterbank


def load_game_config(config_file: str) -> dict:
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def merge_onsets(cur_onsets, stk_note_coords, coords2onsets):
    """ Merge onsets occurring in the same frame """

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
    onsets = np.asarray(onsets, dtype=np.int)

    return onsets, coords


def pad_representation(rep: np.ndarray, onset: np.ndarray, pad_offset: int, pad_value=0) -> (np.ndarray, np.ndarray):
    """Pad representation with zeros at the beginning and the end."""

    rep_out = None

    if rep.ndim == 2:
        rep_out = np.pad(rep, ((0, 0), (pad_offset, pad_offset)), mode='constant', constant_values=pad_value)

    if rep.ndim == 3:
        rep_out = np.pad(rep, ((0, 0), (0, 0), (pad_offset, pad_offset)), mode='constant', constant_values=pad_value)

    onset_out = onset + pad_offset

    return rep_out, onset_out


def load_data_from_dir(score_folder='score', perf_folder='performance', directory='test_sample',
                       real_perf=False):

    data = {}

    for cur_path_score in tqdm.tqdm(glob.glob(os.path.join(directory, score_folder, '*.npz'))):

        cur_path_perf = os.path.join(directory, perf_folder, os.path.split(cur_path_score)[-1])

        song_name = os.path.splitext(os.path.basename(os.path.normpath(cur_path_score)))[0]

        npzfile = np.load(cur_path_score, allow_pickle=True)
        score = (npzfile["sheet"], npzfile["coords"], npzfile['coord2onset'][0])

        if real_perf:
            perf = cur_path_perf.replace('.mid', '.wav')
        else:
            cur_path_perf = cur_path_perf.replace('.npz', '.mid')
            perf = pm.PrettyMIDI(cur_path_perf)

        data[song_name] = {'perf': perf, 'score': score}

    return data


def midi_reset_instrument(midi, id=0):
    """Set all instruments to `id`."""
    for cur_instr in midi.instruments:
        cur_instr.program = id
    return midi


def midi_reset_start(midi):
    """First onset is at position 0.
       Everything is adjusted accordingly.
    """
    for cur_instr in midi.instruments:
        # find index of first note, the first in the array is not necessarily the one with the earliest starting time
        first_idx = np.argmin([n.start for n in cur_instr.notes])
        first_onset = cur_instr.notes[first_idx].start
        for cur_note in cur_instr.notes:
            cur_note.start = max(0, cur_note.start - first_onset)

    return midi


def spectrogram_processor(spec_params):

    """Helper function for our spectrogram extraction."""
    sig_proc = SignalProcessor(num_channels=1, sample_rate=spec_params['sample_rate'])
    fsig_proc = FramedSignalProcessor(frame_size=spec_params['frame_size'], fps=spec_params['fps'])

    spec_proc = FilteredSpectrogramProcessor(filterbank=LogarithmicFilterbank, num_bands=12, fmin=60, fmax=6000,
                                             norm_filters=True, unique_filters=False)
    log_proc = LogarithmicSpectrogramProcessor()

    processor = SequentialProcessor([sig_proc, fsig_proc, spec_proc, log_proc])

    return processor


def wav_to_spec(path_audio: str, spec_params: dict):
    """Extract spectrogram from audio."""

    processor = spectrogram_processor(spec_params)

    spec = processor.process(path_audio).T

    if spec_params.get('norm', False):
        spec = librosa.util.normalize(spec, norm=2, axis=0, threshold=0.01, fill=False)

    return np.expand_dims(spec, 0)


def midi_to_spec_otf(midi: pm.PrettyMIDI, spec_params: dict, sound_font_path=None) -> np.ndarray:
    """MIDI to Spectrogram (on the fly)

       Synthesizes a MIDI with fluidsynth and extracts a spectrogram.
       The spectrogram is directly returned
    """
    processor = spectrogram_processor(spec_params)

    def render_audio(midi_file_path, sound_font):
        """
        Render midi to audio
        """

        # split file name and extention
        name, extention = midi_file_path.rsplit(".", 1)

        # set file names
        audio_file = name + ".wav"

        # audio_file = tempfile.TemporaryFile('w+b')

        # synthesize midi file to audio
        cmd = "fluidsynth -F %s -O s16 -T wav %s %s 1> /dev/null" % (audio_file, sound_font, midi_file_path)

        os.system(cmd)
        return audio_file

    mid_path = os.path.join(tempfile.gettempdir(), str(time.time())+'.mid')

    with open(mid_path, 'wb') as f:
        midi.write(f)

    audio_path = render_audio(mid_path, sound_font=sound_font_path)

    spec = processor.process(audio_path).T

    if spec_params.get('norm', False):
        spec = librosa.util.normalize(spec, norm=2, axis=0, threshold=0.01, fill=False)

    # compute spectrogram
    spec = np.expand_dims(spec, 0)

    os.remove(mid_path)
    os.remove(audio_path)

    return spec


def midi_to_onsets(midi_file: pm.PrettyMIDI, fps: int, instrument_idx=None, unique=True) -> np.ndarray:
    """Extract onsets from a list of midi files. Only returns unique onsets."""

    if instrument_idx is not None:
        # only get onsets from the right hand
        onsets_list = (midi_file.instruments[instrument_idx].get_onsets()*fps).astype(int)
    else:
        # get all unique onsets merged in one list
        onsets_list = (midi_file.get_onsets()*fps).astype(int)

    return np.unique(onsets_list) if unique else onsets_list


def fluidsynth(midi, fs=44100, sf2_path=None):
    """Synthesize using fluidsynth.
    Copied and adapted from `pretty_midi`.

    Parameters
    ----------
    midi : pm.PrettyMidi
    fs : int
        Sampling rate to synthesize at.
    sf2_path : str
        Path to a .sf2 file.
        Default ``None``, which uses the TimGM6mb.sf2 file included with
        ``pretty_midi``.
    Returns
    -------
    synthesized : np.ndarray
        Waveform of the MIDI data, synthesized at ``fs``.
    """
    # If there are no instruments, or all instruments have no notes, return
    # an empty array
    if len(midi.instruments) == 0 or all(len(i.notes) == 0 for i in midi.instruments):
        return np.array([])
    # Get synthesized waveform for each instrument
    waveforms = []
    for i in midi.instruments:
        if len(i.notes) > 0:
            waveforms.append(i.fluidsynth(fs=fs, sf2_path=sf2_path))

    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))

    # Sum all waveforms in
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform

    # Scale to [-1, 1]
    synthesized /= 2**16
    # synthesized = synthesized.astype(np.int16)

    # normalize
    synthesized /= float(np.max(np.abs(synthesized)))

    return synthesized
