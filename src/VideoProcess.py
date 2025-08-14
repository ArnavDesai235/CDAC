from moviepy import VideoFileClip
from config import WAV_FOLDER

def mp4_to_wav(movie_path):
    """
    Extract audio from a video file and save as WAV.
    """
    filename = movie_path.stem
    temp_audio_path = WAV_FOLDER / f"{filename}.temp_audio.wav"

    video_clip = VideoFileClip(str(movie_path))
    video_clip.audio.write_audiofile(str(temp_audio_path), codec="pcm_s16le")
    video_clip.close()

    return temp_audio_path
