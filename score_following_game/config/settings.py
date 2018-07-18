
import os

# set paths
SOUND_FONT_PATH = "sound_fonts/grand-piano-YDP-20160804.sf2"

# get hostname
hostname = os.uname()[1]

# adopted paths
if hostname in ["rechenknecht0.cp.jku.at", "rechenknecht1.cp.jku.at"]:
    SOUND_FONT_PATH = None

elif hostname == "mdhp":
    SOUND_FONT_PATH = "/home/matthias/cp/src/score_following_game/score_following_game/sound_fonts"

elif hostname == "cns":
    SOUND_FONT_PATH = "sound_fonts/grand-piano-YDP-20160804.sf2"
    # SOUND_FONT_PATH = "/home/matthias/.fluidsynth/grand-piano-YDP-20160804.sf2"
    # SOUND_FONT_PATH = "/home/matthias/.fluidsynth/ElectricPiano.sf2"
    # SOUND_FONT_PATH = "/home/matthias/.fluidsynth/YamahaGrandPiano.sf2"
    # SOUND_FONT_PATH = "/home/matthias/.fluidsynth/acoustic_piano_imis_1.sf2"
