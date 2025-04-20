import cv2
import numpy as np
import time


class ActionDetector:
    def __init__(self, model_path):
        """
        Initialize the action detector with the specified model files.

        Args:
            model_path: Path to the neural network model weights file
            config_path: Path to the neural network configuration file
        """
        # Load the pose detection model
        self.pose_net = cv2.dnn.readNetFromTensorflow(model_path)

        # MPII body parts
        # self.body_parts = {
        #     "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        #     "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        #     "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
        #     "Background": 15
        # }
        self.body_parts = {
            0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
            10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
            15: "LEye", 16: "REar", 17: "LEar", 18: "Background", 19: "Chest",
            20: "Head", 21: "Neck", 22: "RHand", 23: "LHand", 24: "RFoot", 25: "LFoot",
            26: "RShoulder", 27: "LShoulder"
        }

        # Define connections between body parts for drawing
        self.pose_pairs = [
            ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
            ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
            ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]
        ]

        # For action detection over time
        self.keypoint_history = []
        self.max_history = 30  # Store last 30 frames of keypoints
        self.last_detection_time = 0
        self.mouth_movement_counter = 0
        self.last_action = "unknown"

    def detect_keypoints(self, frame):
        """
        Detect body keypoints in the given frame.

        Args:
            frame: Image frame to process

        Returns:
            List of detected keypoints
        """
        frame_height, frame_width = frame.shape[:2]

        # Prepare input blob
        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        self.pose_net.setInput(input_blob)

        # Get network output
        output = self.pose_net.forward()

        # Number of keypoints (body parts)
        n_points = output.shape[1]

        # Initialize list for keypoints
        keypoints = []

        # For each body part
        for i in range(n_points):
            
            if i >= len(self.body_parts):
                print(f"Warning: Detected keypoint index {i} exceeds defined body parts.")
                continue
            
            # Confidence map for the current body part
            prob_map = output[0, i, :, :]

            # Find global max location
            _, prob, _, point = cv2.minMaxLoc(prob_map)

            # Scale the point to the original image dimensions
            x = int(frame_width * point[0] / output.shape[3])
            y = int(frame_height * point[1] / output.shape[2])

            # Add keypoint if the confidence is high enough
            if prob > 0.3:
                keypoints.append({
                    "part": list(self.body_parts.keys())[i],
                    "point": (x, y),
                    "confidence": prob
                })

        return keypoints

    def detect_pose_and_action(self, frame):
        """
        Detect body poses and actions in the given frame.

        Args:
            frame: Image frame to process

        Returns:
            Dictionary with keypoints and detected action
        """
        # Detect keypoints in the frame
        keypoints = self.detect_keypoints(frame)

        # Update keypoint history for temporal analysis
        self.keypoint_history.append(keypoints)
        if len(self.keypoint_history) > self.max_history:
            self.keypoint_history.pop(0)

        # Determine action based on keypoints and history
        action = self.determine_action(keypoints)

        return {
            "keypoints": keypoints,
            "action": action
        }

    def determine_action(self, keypoints):
        """
        Determine the action based on detected keypoints and history.

        Args:
            keypoints: List of detected keypoints

        Returns:
            String identifying the detected action
        """
        # Convert keypoints to dictionary for easier access
        keypoint_dict = {kp["part"]: kp["point"] for kp in keypoints}

        # Check if we have enough keypoints to determine an action
        if len(keypoints) < 5:
            return "unknown"

        current_time = time.time()

        # Action 1: Waving Hand (detect rapid motion of wrists)
        if "RWrist" in keypoint_dict and len(self.keypoint_history) > 5:
            # Check right hand waving
            r_wrist_history = [
                next((kp["point"] for kp in frame_kps if kp["part"] == "RWrist"), None)
                for frame_kps in self.keypoint_history[-10:]  # Use last 10 frames
            ]

            # Filter out None values
            r_wrist_history = [point for point in r_wrist_history if point is not None]

            if len(r_wrist_history) >= 3:  # Need at least 3 points to detect motion
                x_diffs = [abs(r_wrist_history[i][0] - r_wrist_history[i - 1][0])
                        for i in range(1, len(r_wrist_history))]

                # If there's significant horizontal movement
                if any(diff > 5 for diff in x_diffs):  # Adjusted threshold
                    return "waving_hand"

        # Check left hand waving similarly
        if "LWrist" in keypoint_dict and len(self.keypoint_history) > 5:
            l_wrist_history = [
                next((kp["point"] for kp in frame_kps if kp["part"] == "LWrist"), None)
                for frame_kps in self.keypoint_history[-10:]
            ]

            # Filter out None values
            l_wrist_history = [point for point in l_wrist_history if point is not None]

            if len(l_wrist_history) >= 3:
                x_diffs = [abs(l_wrist_history[i][0] - l_wrist_history[i - 1][0])
                        for i in range(1, len(l_wrist_history))]

                if any(diff > 5 for diff in x_diffs):  # Adjusted threshold
                    return "waving_hand"
        # Action 2: Turning Back (detect when shoulders and hips are visible but face isn't)
        if ("RShoulder" in keypoint_dict and "LShoulder" in keypoint_dict and
                ("RHip" in keypoint_dict or "LHip" in keypoint_dict)):

            # If shoulders are visible but head isn't detected with good confidence
            head_visible = any(kp["part"] == "Head" and kp["confidence"] > 0.5 for kp in keypoints)

            if not head_visible:
                return "turning_back"

        # Action 3: Talking (detect consistent small movements in head region)
        if "Head" in keypoint_dict and "Neck" in keypoint_dict:
            head_history = [
                next((kp["point"] for kp in frame_kps if kp["part"] == "Head"), None)
                for frame_kps in self.keypoint_history[-10:]
                if any(kp["part"] == "Head" for kp in frame_kps)
            ]

            if len(head_history) >= 5:
                # Check for small vertical movements of the head (possible talking)
                y_diffs = [abs(head_history[i][1] - head_history[i - 1][1])
                           for i in range(1, len(head_history))]

                # Small head movements but not completely still
                if 3 < np.mean(y_diffs) < 10 and len(y_diffs) >= 4:
                    self.mouth_movement_counter += 1
                    if self.mouth_movement_counter > 15:  # Need consistent movement
                        return "talking"
                else:
                    self.mouth_movement_counter = max(0, self.mouth_movement_counter - 1)

        # Action 4: Sleeping/Resting (head tilted or lowered)
        if "Head" in keypoint_dict and "Neck" in keypoint_dict:
            head = keypoint_dict["Head"]
            neck = keypoint_dict["Neck"]

            # Check if head is significantly tilted or lowered
            head_neck_angle = np.degrees(np.arctan2(head[1] - neck[1], head[0] - neck[0]))

            # If head is significantly tilted to either side
            if abs(head_neck_angle) > 30 or head[1] > neck[1] + 10:  # Head lower than it should be
                # Check if the pose has been stable for a while
                head_movement = 0
                if len(self.keypoint_history) > 10:
                    head_history = [
                        next((kp["point"] for kp in frame_kps if kp["part"] == "Head"), None)
                        for frame_kps in self.keypoint_history[-10:]
                        if any(kp["part"] == "Head" for kp in frame_kps)
                    ]

                    if len(head_history) >= 5:
                        movements = [abs(head_history[i][0] - head_history[i - 1][0]) +
                                     abs(head_history[i][1] - head_history[i - 1][1])
                                     for i in range(1, len(head_history))]
                        head_movement = np.mean(movements) if movements else 0

                # Minimal movement indicates sleeping/resting
                if head_movement < 5:
                    return "sleeping"

        # If no specific action is detected, maintain last action for a while to avoid flickering
        if current_time - self.last_detection_time < 1.0 and self.last_action != "unknown":
            return self.last_action

        # Default: unknown action
        self.last_action = "unknown"
        self.last_detection_time = current_time
        return "unknown"

    def draw_pose(self, frame, keypoints):
        """
        Draw keypoints and connections on the frame.

        Args:
            frame: Image frame to draw on
            keypoints: List of detected keypoints
        """
        # Draw keypoints
        for keypoint in keypoints:
            if "point" in keypoint:
                cv2.circle(frame, keypoint["point"], 5, (0, 0, 255), -1)

        # Draw connections between keypoints
        for pair in self.pose_pairs:
            part_a = pair[0]
            part_b = pair[1]

            # Check if both parts exist in the keypoints
            if part_a in [kp["part"] for kp in keypoints] and \
                    part_b in [kp["part"] for kp in keypoints]:
                point_a = next(kp["point"] for kp in keypoints if kp["part"] == part_a)
                point_b = next(kp["point"] for kp in keypoints if kp["part"] == part_b)

                # Draw the connection
                cv2.line(frame, point_a, point_b, (255, 0, 0), 2)