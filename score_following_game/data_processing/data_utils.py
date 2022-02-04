
import librosa
import yaml

import numpy as np

from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor, \
    LogarithmicFilterbank


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
