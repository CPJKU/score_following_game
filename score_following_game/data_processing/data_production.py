
import glob
import numpy as np
import os
import random
import time

from collections import deque
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from score_following_game.data_processing.song import create_song
from score_following_game.data_processing.utils import load_data_from_dir


class SongCache(object):

    def __init__(self, cache_size=50):

        self.cache = deque(maxlen=cache_size)
        self.song_history = {}
        self.queue = []

    def append(self, value):
        self.cache.append(value)

    def get(self):
        return self.cache

    def get_maxlen(self):
        return self.cache.maxlen

    def get_random(self):

        song = np.random.choice(self.cache)
        self._update_history(song)

        return song

    def get_elem(self, idx):

        song = self.cache[idx]
        self._update_history(song)

        return song

    def get_history(self):
        return self.song_history

    def _update_history(self, song):
        song_name = song.get_song_name()
        if song_name in self.song_history:
            self.song_history[song_name] += 1
        else:
            self.song_history[song_name] = 1
        return song


class SongProducer(Process):

    def __init__(self, cache, config: dict, score_folder='score', perf_folder='performance',
                 directory='test_sample', real_perf=False):
        Process.__init__(self)

        self.cache_size = cache.get_maxlen()
        self.cache = cache
        self.directory = directory
        self.config = config
        self.score_folder = score_folder
        self.perf_folder = perf_folder

        # load raw data into memory
        print('Load data to memory...')
        self.raw_data = load_data_from_dir(score_folder=self.score_folder, perf_folder=self.perf_folder,
                                           directory=self.directory, real_perf=real_perf)

        self.default_sf_path = config['default_sf']

        self.real_perf = real_perf

        self.fill_cache()

    def run(self):

        while True:
            # add a random song to cache
            self.cache.append(self._load_random_song())
            time.sleep(random.random())

    def fill_cache(self):
        """Initial cache filling with `self.cache_size` songs"""

        print('Initial fill of cache')
        for i in range(self.cache_size):
            print("Song {}/{}".format(i, self.cache_size), end="\r")
            self.cache.append(self._load_random_song())

    def _load_random_song(self):

        song_name = random.choice(list(self.raw_data.keys()))

        song = create_song(self.config, song_name, self.raw_data[song_name]['score'],
                           self.raw_data[song_name]['perf'], default_sf_path=self.default_sf_path)

        return song


def create_song_cache(cache_size=50):
    BaseManager.register('SongCache', SongCache)
    manager = BaseManager()
    manager.start()

    cache = manager.SongCache(cache_size)
    return cache


def create_song_producer(cache, config, score_folder='score', perf_folder='performance',
                         directory='test_sample', real_perf=False):

    producer = SongProducer(cache, config=config, perf_folder=perf_folder, score_folder=score_folder,
                            directory=directory, real_perf=real_perf)
    return producer
