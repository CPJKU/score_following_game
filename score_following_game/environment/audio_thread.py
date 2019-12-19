import threading

import numpy as np


class AudioThread(threading.Thread):

    def __init__(self, path_to_audio, fps):
        threading.Thread.__init__(self)
        self.path_to_audio = path_to_audio
        self.frameRate = 1
        self.frame = 0
        self.fps = fps
        self.waveform = None
        self.py_audio = None
        self.stream = None

    def get_current_position(self):
        return self.frame / float(self.frameRate)

    def get_current_spec_position(self):
        """
        :return: current position (frame index) in spectrogram
        """
        return int(np.ceil((self.frame / float(self.frameRate)) * self.fps))

    def run(self):
        import wave
        import pyaudio

        # adapted from https://people.csail.mit.edu/hubert/pyaudio/docs/#example-callback-mode-audio-i-o
        self.waveform = wave.open(self.path_to_audio, 'rb')
        self.frameRate = self.waveform.getframerate()

        # instantiate PyAudio
        self.py_audio = pyaudio.PyAudio()

        # define callback
        def callback(in_data, frame_count, time_info, status):
            self.frame += frame_count
            data = self.waveform.readframes(frame_count)
            return data, pyaudio.paContinue

        # open stream using callback
        self.stream = self.py_audio.open(format=self.py_audio.get_format_from_width(self.waveform.getsampwidth()),
                                         channels=self.waveform.getnchannels(),
                                         rate=self.frameRate,
                                         output=True,
                                         stream_callback=callback)

        # start the stream
        self.stream.start_stream()

    def end_stream(self):

        # stop stream
        self.stream.stop_stream()
        self.stream.close()
        self.waveform.close()

        # close PyAudio
        self.py_audio.terminate()

    def still_playing(self):
        return self.stream.is_active()
