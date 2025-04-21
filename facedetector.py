import cv2
import dlib
import time
import serial
import serial.tools.list_ports

# Function to auto-detect Arduino port
'''
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        try:
            # Attempt to open the port
            ser = serial.Serial(port.device, 9600, timeout=1)
            time.sleep(2)  # Wait for connection to stabilize
            ser.write(b'0')  # Send a test byte to check if Arduino responds
            ser.close()  # Close the test connection
            return port.device  # Return the valid port
        except (serial.SerialException, OSError):
            continue
    return None

# Initialize serial communication with Arduino
port = find_arduino_port()
if port is None:
    print("Error: Could not find an Arduino. Please check the connection.")
    exit()
try:
    ser = serial.Serial(port, 9600, timeout=1)
    time.sleep(2)  # Wait for the serial connection to initialize
except serial.SerialException as e:
    print(f"Error opening serial port {port}: {e}")
    exit()
    '''

# Load face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye_points):
    vertical_1 = ((eye_points[1].x - eye_points[5].x) ** 2 + (eye_points[1].y - eye_points[5].y) ** 2) ** 0.5
    vertical_2 = ((eye_points[2].x - eye_points[4].x) ** 2 + (eye_points[2].y - eye_points[4].y) ** 2) ** 0.5
    horizontal = ((eye_points[0].x - eye_points[3].x) ** 2 + (eye_points[0].y - eye_points[3].y) ** 2) ** 0.5
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Function to detect closed eyes
def detect_closed_eyes(frame, landmarks):
    left_eye_points = [landmarks.part(i) for i in range(36, 42)]
    right_eye_points = [landmarks.part(i) for i in range(42, 48)]
    left_ear = eye_aspect_ratio(left_eye_points)
    right_ear = eye_aspect_ratio(right_eye_points)
    ear = (left_ear + right_ear) / 2.0
    for i in range(36, 48):
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (255, 255, 255), -1)
    return ear

# Function to detect yawning
def detect_yawn(frame, landmarks):
    mouth_points = [landmarks.part(i) for i in range(48, 68)]
    if len(mouth_points) == 20:
        mouth_width = abs(mouth_points[6].x - mouth_points[0].x)
        mouth_height = abs(mouth_points[3].y - mouth_points[9].y)
        aspect_ratio = mouth_height / mouth_width if mouth_width != 0 else 0
        for i in range(20):
            cv2.circle(frame, (mouth_points[i].x, mouth_points[i].y), 2, (255, 255, 255), 2)
            cv2.line(frame, (mouth_points[i].x, mouth_points[i].y), (mouth_points[(i+1)%20].x, mouth_points[(i+1)%20].y), (255, 255, 255), 2)
        return aspect_ratio
    return None

# Open video capture for webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    ser.close()
    exit()

# Set the window size
cv2.namedWindow('Drowsiness and Yawn Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Drowsiness and Yawn Detection', 800, 700)

# Initialize variables for eye detection
closed_eye_start = None
closed_eye_alert_threshold = 0.5
eye_closed_count = 0
yawn_count = 0

# Process webcam frames until the user exits
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        landmarks = landmark_predictor(gray, face)
        ear = detect_closed_eyes(frame, landmarks)
        ear_threshold = 0.25
        if ear < ear_threshold:
            if closed_eye_start is None:
                closed_eye_start = time.time()
            else:
                closed_eye_duration = time.time() - closed_eye_start
                if closed_eye_duration >= closed_eye_alert_threshold:
                    cv2.putText(frame, "Drowsiness Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    eye_closed_count += 1
                    '''
                    # Send '1' to Arduino
                    try:
                        ser.write(b'1')
                    except serial.SerialException as e:
                        print(f"Error writing to serial port: {e}")'''
        else:
            closed_eye_start = None

        aspect_ratio = detect_yawn(frame, landmarks)
        if aspect_ratio is not None:
            yawn_threshold = 1.5
            if aspect_ratio > yawn_threshold:
                cv2.putText(frame, "Yawn Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                yawn_count += 1

    cv2.imshow('Drowsiness and Yawn Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
ser.close()
cv2.destroyAllWindows()
