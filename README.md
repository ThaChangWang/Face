# Face

Frame Manipulation using dlib and cv2.

Features

videoDownload.py
  - Download a youtube video that you want to work with as both an audio file and a video file.
  
face.py
  - Find face in video and find all facial points for each frame.
  - Specify which facial feature you want in the command line args.
  - Returns images containing that facial feature withc everything else masked out.
  
dummy.py
  - Find face in video and find all facial points for each frame.
  - See which folders exist in directory (ex. "left_eyebrow", "mouth") and paste them on top of each face in the video.
  - Returns frames with the facial features pasted over the faces of the video.
  
imageCombiner.py
  - Combines all frames in folder.
  
addAudio.py
  - Adds audio to video.
  
