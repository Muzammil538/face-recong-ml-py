import cv2
import numpy as np
import face_recognition
from pathlib import Path
import os

from actions.action_detector import ActionDetector


class FaceDetectionRecognition:
    def __init__(self, known_faces_folder):
        """
        Initialize the face detection and recognition system.

        Args:
            known_faces_folder: Path to folder containing known face images
        """
        self.known_face_encodings = []
        self.known_face_names = []

        # Create folder if it doesn't exist
        os.makedirs(known_faces_folder, exist_ok=True)

        self.load_known_faces(known_faces_folder)

        # Initialize pose/action detection
        pose_model_path = os.path.abspath('tf-pose-estimation/models/graph/mobilenet_thin_432x368/graph_opt.pb')

        # Initialize action detector
        if os.path.exists(pose_model_path):
            self.action_detector = ActionDetector(pose_model_path)
            self.pose_enabled = True
        else:
            print("Warning: Pose detection model files not found. Pose detection disabled.")
            self.action_detector = None
            self.pose_enabled = False

    def load_known_faces(self, known_faces_folder):
        """
        Load known faces from the specified folder.
        Each subfolder name is used as the person's name.

        Args:
            known_faces_folder: Path to folder containing subfolders of person faces
        """
        known_faces_path = Path(known_faces_folder)

        # Check if folder exists and has content
        if not known_faces_path.exists() or not any(known_faces_path.iterdir()):
            print(f"Known faces folder empty or not found: {known_faces_folder}")
            print("Please add face images organized in person-named subfolders.")
            return

        # Process each person's subfolder
        for person_folder in known_faces_path.iterdir():
            if person_folder.is_dir():
                person_name = person_folder.name
                print(f"Loading face data for: {person_name}")

                # Process each image in the person's folder
                face_count = 0
                image_files = list(person_folder.glob("*.jpg")) + list(person_folder.glob("*.jpeg")) + list(
                    person_folder.glob("*.png"))

                if not image_files:
                    print(f"  - No images found for {person_name}")
                    continue

                for img_path in image_files:
                    try:
                        # Load image and find face encodings
                        face_image = face_recognition.load_image_file(img_path)
                        face_encodings = face_recognition.face_encodings(face_image)

                        # If a face was found, add it to known faces
                        if face_encodings:
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(person_name)
                            face_count += 1
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")

                print(f"  - Added {face_count} face samples for {person_name}")

        if self.known_face_encodings:
            print(
                f"Loaded {len(self.known_face_encodings)} total face samples for {len(set(self.known_face_names))} people")
        else:
            print("No face samples were loaded. Please check your images.")

    def recognize_faces(self, frame):
        """
        Recognize faces in the given frame.

        Args:
            frame: Image frame to process

        Returns:
            List of (name, bounding box) tuples for each detected face
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_faces = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            name = "Unknown"
            confidence = 0

            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)

                # Use the known face with the smallest distance
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]

            # Convert back to original frame size
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            recognized_faces.append({
                "name": name,
                "box": (left, top, right, bottom),
                "confidence": confidence if name != "Unknown" else 0
            })

        return recognized_faces

    def process_frame(self, frame):
        """
        Process a frame: detect faces, recognize known faces, and detect poses.

        Args:
            frame: Image frame to process

        Returns:
            Processed frame with annotations and detection results
        """
        # Make a copy of the frame to draw on
        display_frame = frame.copy()

        # Recognize faces
        recognized_faces = self.recognize_faces(frame)

        # Draw face boxes and names
        for face in recognized_faces:
            left, top, right, bottom = face["box"]
            name = face["name"]
            confidence = face["confidence"]

            # Draw box
            cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw name and confidence
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(display_frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Detect poses and actions
        if self.pose_enabled and self.action_detector:
            pose_results = self.action_detector.detect_pose_and_action(frame)

            # Draw pose keypoints and connections
            self.action_detector.draw_pose(display_frame, pose_results["keypoints"])

            # Display detected action
            if "action" in pose_results:
                cv2.putText(display_frame, f"Action: {pose_results['action']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            pose_results = {"keypoints": [], "action": "unknown"}

        return display_frame, recognized_faces, pose_results