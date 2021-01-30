from pytube import YouTube
from moviepy.editor import *
import sys

yt = YouTube("https://www.youtube.com/watch?v=XbDyn-2_xIc")

thisStream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
thisStream.download(filename=sys.argv[1])

video = VideoFileClip(sys.argv[1] + ".mp4").subclip(890, 927)
video.write_videofile("facevid.mp4")