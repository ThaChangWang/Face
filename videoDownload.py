from pytube import YouTube

yt = YouTube("https://www.youtube.com/watch?v=-x-GC37SGkE")

thisStream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
thisStream.download(filename="mitch")