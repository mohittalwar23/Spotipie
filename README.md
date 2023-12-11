# SPOTIPE - Emotion Recognition and Hand Gesture Controlled Spotify Assistant

This project combines emotion recognition using facial landmarks and hand gesture control to interact with the Spotify API.

## Reference from 
- Rattasart Sakunrat
- Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

## Introduction

This project integrates emotion recognition based on facial landmarks with hand gesture control for interacting with the Spotify API. It uses the Mediapipe library for facial landmark detection and the CVZone library for hand gesture recognition.

## Features

- Emotion recognition using facial landmarks.
- Hand gesture control for Spotify playback.
- Automatic playlist selection based on detected mood.

## Installation


- Clone the repository.

```plaintext
  git clone https://github.com/mohittalwar23/Spotipie
```

- Install the Required packages.

```plaintext
  pip install -r requirements2.txt
```

- Set up the Spotify API credentials:
  - You will need a Spotify premium account for this.
  - Create a Spotify Developer account at https://developer.spotify.com/ and create a new app.
  - Obtain the CLIENT_ID, CLIENT_SECRET, and set the redirect_uri to http://localhost:8080 in the SpotifyOAuth initialization.

## Dependencies

Ensure you have the following dependencies installed:

```plaintext
  cv2
  csv
  copy
  itertools
  numpy
  mediapipe
  model (custom module)
  cvzone
  time
  spotipy
  pyttsx3
  threading
```

## Author 

Mohit Talwar

## References

Some instances in the first draft were referred from:

- [Spotify-RFID-Record-Player](https://github.com/talaexe/Spotify-RFID-Record-Player)

In the second draft, an attempt was made to replicate emotion detection with CNN:

- [Emotion Detection with CNN](https://github.com/datamagic2020/Emotion_detection_with_CNN/tree/main)

However, we switched to using MEDIAPIPE in the third draft.

The entire emotion detection part used in our code is referenced from:

- [Facial Emotion Recognition using Mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)

## License

hand-gesture-recognition-using-mediapipe is under [Apache V2 License](https://github.com/mohittalwar23/Spotipie/blob/main/LICENSE)



















