import cv2
from random import randrange as r
import serial
import serial.tools.list_ports
import time

def find_arduino_port():
    """Automatically detect Arduino serial port."""
    arduino_ports = []
    for port in serial.tools.list_ports.comports():
        if 'Arduino' in port.description or 'CH340' in port.description or 'USB Serial' in port.description:
            arduino_ports.append(port.device)
        if port.vid in [0x2341, 0x0403, 0x1A86]:
            arduino_ports.append(port.device)

    if not arduino_ports:
        print("No Arduino detected. Continuing without serial communication.")
        return None
    if len(arduino_ports) > 1:
        print(f"Multiple Arduino ports found: {arduino_ports}. Using the first one.")

    return arduino_ports[0]

# Initialize serial communication
ser = None
port = find_arduino_port()
if port:
    try:
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {port}")
    except serial.SerialException as e:
        print(f"Failed to connect to {port}: {e}")
        ser = None
else:
    print("No serial connection established.")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("Error loading cascades")
    if ser:
        ser.close()
    exit(1)

# Start webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Webcam error")
    if ser:
        ser.close()
    exit(1)

while True:
    success, frame = webcam.read()
    if not success:
        print("Error reading webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    eyes_open = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (r(0,256), r(0,256), r(0,256)), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)
        if len(eyes) > 0:
            eyes_open = True
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Debug print
    print("Sending:", int(eyes_open))

    # Send signal to Arduino
    if ser:
        try:
            ser.write(f"{int(eyes_open)}\n".encode())  # include newline
        except serial.SerialException as e:
            print(f"Serial write failed: {e}")

    # Show video
    cv2.imshow('Eye Detector', frame)

    # Press 's' or 'S' to exit
    key = cv2.waitKey(1)
    if key in [ord('s'), ord('S')]:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
if ser:
    try:
        ser.close()
    except serial.SerialException:
        pass

