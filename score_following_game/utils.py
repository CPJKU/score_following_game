import cv2
import numpy as np
import os
import shutil
import soundfile as sf


def write_video(images, fn_output='output.mp4', frame_rate=20, overwrite=False):
    """Takes a list of images and interprets them as frames for a video.

    Source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    """
    height, width, _ = images[0].shape

    if overwrite:
        if os.path.exists(fn_output):
            os.remove(fn_output)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(fn_output, fourcc, frame_rate, (width, height))

    for cur_image in images:
        frame = cv2.resize(cur_image, (width, height))
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()

    return fn_output


def mux_video_audio(path_video, path_audio, path_output='output_audio.mp4'):
    """Use FFMPEG to mux video with audio recording."""
    from subprocess import check_call

    check_call(["ffmpeg", "-y", "-i", path_video, "-i", path_audio, "-shortest", path_output])


def render_video(observation_images, pool, fps=20, mux_audio=True, real_perf=False, video_path='videos'):

    if not os.path.isdir(video_path):
        os.mkdir(video_path)

    if mux_audio:
        fn_audio = 'tmp.wav'

        if real_perf == 'wav':
            # copy over wav file
            shutil.copy(pool.curr_song.path_perf, fn_audio)
        else:
            # get synthesized MIDI as WAV
            perf_audio, fs = pool.get_current_perf_audio_file()
            sf.write(fn_audio, perf_audio, fs)

        # frame rate video is now based on the piano roll's frame rate
        path_video = write_video(observation_images, fn_output=pool.get_current_song_name() + '.mp4',
                                 frame_rate=fps, overwrite=True)

        # mux video and audio with ffmpeg
        mux_video_audio(path_video, fn_audio, path_output=os.path.join(video_path,
                                                                       pool.get_current_song_name() + '_audio.mp4'))

        # clean up
        os.remove(fn_audio)
        os.remove(path_video)
    else:
        write_video(observation_images, frame_rate=1, overwrite=True)


def get_opencv_bar(value, bar_heigth=500, min_value=0, max_value=11, color=(255, 255, 0), title=None):

    value_coord = bar_heigth - int(float(bar_heigth - 20) * value / max_value) + 20
    value_coord = np.clip(value_coord, min_value, bar_heigth - 1)

    bar_img_bgr = np.zeros((bar_heigth, 100, 3), np.uint8)
    cv2.line(bar_img_bgr, (0, value_coord), (bar_img_bgr.shape[1] - 1, value_coord), color, 5)

    # write current speed to observation image
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text = "%.2f" % value
    text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
    text_org = (100 - text_size[0], value_coord - 6)
    cv2.putText(bar_img_bgr, text, text_org, fontFace=font_face, fontScale=0.6, color=color, thickness=1)

    if title is not None:
        text_size = cv2.getTextSize(title, fontFace=font_face, fontScale=0.6, thickness=1)[0]
        text_org = (100 // 2 - text_size[0] // 2, 20)
        cv2.putText(bar_img_bgr, title, text_org, fontFace=font_face, fontScale=0.6, color=color, thickness=1)

    return bar_img_bgr
