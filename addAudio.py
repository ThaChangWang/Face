# Import everything needed to edit video clips 
from moviepy.editor import *
from pydub import AudioSegment
import os


# loading video dsa gfg intro video 
clip = VideoFileClip("final.avi") 
soundClip = AudioSegment.from_mp3("aud.mp3")
soundClip.export("aud.wav", format="wav")

os.system("ffmpeg -i final.avi -i aud.wav -vcodec copy -acodec copy finalwaud.avi")


# loading audio file 
#audioclip = AudioFileClip("aud.wav")

# adding audio to the video clip 
#videoclip = clip.set_audio(audioclip) 

# showing video clip 
#videoclip.write_videofile("finalwaud.avi") 
