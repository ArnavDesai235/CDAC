from moviepy import VideoFileClip
from pydub import AudioSegment
#input video path as movie
def mp4_to_wav(movie):
    
    filename = movie.split("/")[-1].split(".")[0]

    # Load the video file
    video_clip = VideoFileClip(movie)

    # Export the audio to a temporary file
    #Enter your audio parth
    temp_audio_path = ""
    video_clip.audio.write_audiofile(temp_audio_path, codec="pcm_s16le")

    # Close the video file to release resources
    video_clip.close()

    # Return the generated WAV file path
    return temp_audio_path

