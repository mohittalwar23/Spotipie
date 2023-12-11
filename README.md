# SPOTIPE - Emotion Recognition and Hand Gesture Controlled Spotify Assistant

This project combines emotion recognition using facial landmarks and hand gesture control to interact with the Spotify API.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Reference](#reference)

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
  You will need a Spotify premium account for this.
  Create a Spotify Developer account at https://developer.spotify.com/ and create a new app.
  Obtain the CLIENT_ID, CLIENT_SECRET, and set the redirect_uri to "http://localhost:8080" in the SpotifyOAuth initialization.

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














