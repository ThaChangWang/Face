from pytube import YouTube
from moviepy.editor import *
import sys

yt = YouTube("https://www.youtube.com/watch?v=kQoxbDOWjf4")

thisStream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
thisStream.download(filename="yt")

video = VideoFileClip("yt.mp4")

video.write_videofile("vid.mp4")
video.audio.write_audiofile("aud.mp3")