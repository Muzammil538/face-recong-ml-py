import os
import cv2
import time
import ssl
from pathlib import Path

# Import our modules
from utils.setup import setup_models, create_sample_directory
from models.face_action_detector import FaceDetectionRecognition


def main():
    """
    Main function to run the face and action detection system.
    """
    print("Setting up face and action detection system...")

    # Set context for SSL
    ssl._create_default_https_context = ssl._create_unverified_context

    # Setup required models
    models_downloaded = setup_models()

    # if models_downloaded:
    #     net = cv2.dnn.readNetFromCaffe(
    #     "models/pose_deploy_linevec.prototxt",
    #     "models/pose_iter_440000.caffemodel"
    #     )


    # Create sample directory structure if needed
    create_sample_directory()

    # Path to folder containing known faces
    known_faces_folder = "./image_data"

    # Initialize the face detection and recognition system
    system = FaceDetectionRecognition(known_faces_folder)

    # Check if we have any known faces
    if not system.known_face_encodings:
        print("\nWarning: No face samples loaded. Face recognition will identify all faces as 'Unknown'.")
        print("To use face recognition, add face images to subfolders in the 'known_faces' directory.")
        print("Each subfolder name will be used as the person's identity.\n")

    if not models_downloaded and not system.pose_enabled:
        print("\nWarning: Pose detection models could not be downloaded and are not available.")
        print("Action detection will be disabled.\n")

    # Open webcam
    print("Opening webcam...")
    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(1)

    if not video_capture.isOpened():
        print("Could not open webcam. Please check your camera connection.")
        return

    print("System ready! Press 'q' to quit")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame from camera. Check your webcam connection.")
            break

        # Process the frame
        start_time = time.time()
        display_frame, recognized_faces, pose_results = system.process_frame(frame)
        processing_time = time.time() - start_time

        # Display FPS
        fps = 1.0 / processing_time if processing_time > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Face & Action Detection', display_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()