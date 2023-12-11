# Import necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify API credentials
DEVICE_ID = "your_device_id"
CLIENT_ID = "your_client_id"
CLIENT_SECRET = "your_client_secret"

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri="http://localhost:8080",
    scope="user-read-playback-state,user-modify-playback-state"
))

# Command execution control variables
cooldown_duration = 5
last_command_time = 0
last_action = None
display_timer = 0

# Function to execute Spotify commands
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
        elif command == 'pause':
            sp.pause_playback(device_id=device_id)
        elif command == 'next':
            if current_state is not None and current_state["is_playing"]:
                sp.next_track(device_id=device_id)
        elif command == 'previous':
            if current_state is not None and current_state["is_playing"]:
                # Check if we are not at the beginning of the playlist
                if current_state["progress_ms"] > 3000:  # Assuming that 3000 ms (3 seconds) is a reasonable threshold
                    sp.previous_track(device_id=device_id)

        # Update control variables
        last_command_time = current_time
        last_action = command
        display_timer = current_time + 2  # Display the text for 2 seconds

    except Exception as e:
        print(f"An error occurred: {e}")

# Main loop for video capture and hand tracking
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        centerPoint1 = hand1["center"]
        handType1 = hand1["type"]

        fingers1 = detector.fingersUp(hand1)

        # Check hand gestures and execute corresponding commands
        if handType1 == "Right":
            if fingers1[0] == 0 and fingers1[1] == 1 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('play', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 1 and fingers1[2] == 1 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('next', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 1:
                execute_command('previous', DEVICE_ID)
            if fingers1[0] == 0 and fingers1[1] == 0 and fingers1[2] == 0 and fingers1[3] == 0 and fingers1[4] == 0:
                execute_command('pause', DEVICE_ID)

    # Display the text for a certain duration
    if time.time() < display_timer:
        cv2.putText(img, f"Command: {last_action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
