import cv2 as cv 
import csv
import copy
import itertools
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
from cvzone.HandTrackingModule import HandDetector
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pyttsx3
import threading

# Emotion Recognition Setup
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

use_brect = True
# Hand Gesture and Spotify Setup
cap = cv.VideoCapture(0)
hand_detector = HandDetector(detectionCon=0.8, maxHands=2)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0

# Spotify API credentials
DEVICE_ID = "your_device_id"
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri="http://localhost:8080",
    scope="user-read-playback-state,user-modify-playback-state,playlist-read-private"
))

engine = pyttsx3.init()

cooldown_duration = 5
last_command_time = 0
last_action = None
display_timer = 0
repeat_mood_detection = False
tts_lock = threading.Lock()  # Lock for thread safety

def execute_command(command, device_id):
    global last_command_time, last_action, display_timer
    current_time = time.time()

    # Check cooldown and repeated commands
    if (current_time - last_command_time < cooldown_duration) and (last_action == command):
        return

    try:
        # Check the current playback state
        current_state = sp.current_playback()
        if current_state is not None and current_state["is_playing"]:
            # If the playback state matches the command, skip the command
            if (command == 'play' and current_state["is_playing"]) or \
               (command == 'pause' and not current_state["is_playing"]):
                return

        # Execute the command based on the input
        if command == 'play':
            sp.start_playback(device_id=device_id)
            run_tts_in_thread("Playing !!")
        elif command == 'pause':
            sp.pause_playback(device_id=device_id)
            run_tts_in_thread("Paused !!")
        elif command == 'next':
            if current_state is not None and current_state["is_playing"]:
                sp.next_track(device_id=device_id)
                run_tts_in_thread("Playing Next track")
        elif command == 'previous':
            if current_state is not None and current_state["is_playing"]:
                # Check if we are not at the beginning of the playlist
                if current_state["progress_ms"] > 3000:  # Assuming that 3000 ms (3 seconds) is a reasonable threshold
                    sp.previous_track(device_id=device_id)
                    run_tts_in_thread("Playing Previous track")

        last_command_time = current_time
        last_action = command
        display_timer = current_time + 2  # Display the text for 2 seconds

    except Exception as e:
        print(f"An error occurred: {e}")

def run_tts_in_thread(message):
    threading.Thread(target=execute_tts, args=(message,), daemon=True).start()

def execute_tts(message):
    with tts_lock:
        engine.say(message)
        engine.runAndWait()

def play_playlist(mood):
    try:
        playlists = sp.current_user_playlists()
        for playlist in playlists['items']:
            if mood.lower() in playlist['name'].lower():
                sp.start_playback(device_id=DEVICE_ID, context_uri=playlist['uri'])
                run_tts_in_thread(f"Playing {playlist['name']} playlist for your {mood.lower()} mood.")
                return True

        run_tts_in_thread(f"Sorry, I couldn't find a playlist for your {mood.lower()} mood.")
        return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def determine_mood():
    default_mood = "Neutral"
    detected_mood = None

    run_tts_in_thread("Let's determine your mood. Please wait.")

    for _ in range(60):
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Emotion classification
                facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                detected_mood = keypoint_classifier_labels[facial_emotion_id]

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                        debug_image,
                        brect,
                        detected_mood)

                cv.imshow('Mood Detection', debug_image)
                cv.waitKey(1)

    if detected_mood:
        mood = detected_mood
        run_tts_in_thread(f"Your current mood is {mood.lower()}.")

    return detected_mood or default_mood


def greet_and_start():
    run_tts_in_thread("Hello! Welcome to your personalized assistant.")
    mood = determine_mood()
    play_playlist(mood)

# Initial greeting and mood determination
greet_and_start()

def detect_emotion():
    global repeat_mood_detection
    mood = determine_mood()
    play_playlist(mood)
    repeat_mood_detection = False

while True:
    success, img = cap.read()
    hands, img = hand_detector.findHands(img, draw=True)

    # Check for hand gestures (continuous)
    for hand in hands:
        lm_list = hand["lmList"]
        center_point = lm_list[9]  # Assuming index 9 is the center of the palm
        fingers1 = hand_detector.fingersUp(hand)
        hand_type = hand["type"]

        if hand_type == "Right":
            if fingers1[0] == 0 and fingers1[1] == 1 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('play', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('next', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 1:
                execute_command('previous', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('pause', DEVICE_ID)

        if hand_type == "Left":
            if fingers1[0] == 1 and fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
                detect_emotion()
                
            
        

    # Display the text for a certain duration
    if time.time() < display_timer:
        cv.putText(img, f"Command: {last_action}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imshow("Assistant", img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

 
cap.release()
cv.destroyAllWindows()
