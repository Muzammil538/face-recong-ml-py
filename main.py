import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import pickle
import time

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection System")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        
        # Variables
        self.faces_folder = ""
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.name_mapping = {}  # Maps label numbers to names
        self.camera_active = False
        self.current_frame = None
        self.detection_thread = None
        self.display_thread = None
        self.model_trained = False
        self.frame_queue = []
        self.last_detection_time = 0
        self.detection_interval = 0.1  # seconds between detections

        # Create UI elements
        self.create_widgets()
        
        # Add these lines to initialize eye and mouth detectors
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        
        # Initialize camera
        self.cap = None
        
    def create_widgets(self):
        # Create frames
        self.top_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.video_frame = tk.Frame(self.root, bg="#333333")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.bottom_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Top frame controls
        self.folder_btn = tk.Button(self.top_frame, text="Select Faces Folder", command=self.select_faces_folder, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.folder_btn.pack(side=tk.LEFT, padx=5)
        
        self.folder_label = tk.Label(self.top_frame, text="No folder selected", bg="#f0f0f0", font=("Arial", 12))
        self.folder_label.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = tk.Button(self.top_frame, text="Train Model", command=self.train_model, bg="#2196F3", fg="white", font=("Arial", 12))
        self.train_btn.pack(side=tk.RIGHT, padx=5)
        
        # Video display
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Bottom controls
        self.start_btn = tk.Button(self.bottom_frame, text="Start Camera", command=self.toggle_camera, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.start_btn.pack(side=tk.LEFT, padx=5)

        # Performance slider
        self.sensitivity_label = tk.Label(self.bottom_frame, text="Detection Speed:", bg="#f0f0f0", font=("Arial", 12))
        self.sensitivity_label.pack(side=tk.LEFT, padx=10)
        
        self.sensitivity_slider = tk.Scale(self.bottom_frame, from_=1, to=10, orient=tk.HORIZONTAL, length=100,
                                          command=self.update_sensitivity)
        self.sensitivity_slider.set(5)  # Default middle value
        self.sensitivity_slider.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(self.bottom_frame, text="Status: Ready", bg="#f0f0f0", font=("Arial", 12))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.found_label = tk.Label(self.bottom_frame, text="", bg="#f0f0f0", font=("Arial", 12, "bold"), fg="blue")
        self.found_label.pack(side=tk.RIGHT, padx=10)

    def update_sensitivity(self, value):
        value = int(value)
        # Map slider value (1-10) to detection interval (0.5s to 0.05s)
        self.detection_interval = 0.5 - ((value - 1) * 0.05)
    
    def select_faces_folder(self):
        folder = filedialog.askdirectory(title="Select Folder with Face Images")
        if folder:
            self.faces_folder = folder
            self.folder_label.config(text=f"Selected: {os.path.basename(folder)}")
    
    def train_model(self):
        if not self.faces_folder:
            messagebox.showerror("Error", "Please select faces folder first")
            return
        
        self.status_label.config(text="Status: Training model...")
        self.root.update()
        
        try:
            faces = []
            labels = []
            self.name_mapping = {}
            label_counter = 0
            
            # Get image files from the selected folder
            image_files = [f for f in os.listdir(self.faces_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected folder")
                self.status_label.config(text="Status: No images found")
                return
            
            # Process each image
            for image_file in image_files:
                name = os.path.splitext(image_file)[0]  # Get filename without extension
                image_path = os.path.join(self.faces_folder, image_file)
                
                # Add name to mapping if not exists
                if name not in self.name_mapping.values():
                    self.name_mapping[label_counter] = name
                    current_label = label_counter
                    label_counter += 1
                else:
                    # Find the existing label for this name
                    current_label = [k for k, v in self.name_mapping.items() if v == name][0]
                
                # Load image in grayscale
                img = cv2.imread(image_path)
                if img is None:
                    messagebox.showwarning("Warning", f"Could not read image {image_file}")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces - with optimized parameters
                detected_faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                if len(detected_faces) > 0:
                    for (x, y, w, h) in detected_faces:
                        face_sample = gray[y:y+h, x:x+w]
                        # Add size check to prevent very small faces
                        if w > 60 and h > 60:
                            # Equalize histogram for better recognition in varying lighting
                            face_sample = cv2.equalizeHist(face_sample)
                            faces.append(face_sample)
                            labels.append(current_label)
                    
                    self.status_label.config(text=f"Status: Processed {name}")
                    self.root.update()
                else:
                    messagebox.showwarning("Warning", f"No face detected in {image_file}")
            
            if faces:
                # Train the model
                self.face_recognizer.train(faces, np.array(labels))
                self.model_trained = True
                
                # Save the model and name mapping
                model_dir = os.path.join(self.faces_folder, "model")
                os.makedirs(model_dir, exist_ok=True)
                self.face_recognizer.write(os.path.join(model_dir, "face_model.yml"))
                with open(os.path.join(model_dir, "name_mapping.pkl"), 'wb') as f:
                    pickle.dump(self.name_mapping, f)
                
                self.status_label.config(text=f"Status: Trained with {len(faces)} faces")
            else:
                self.status_label.config(text="Status: No faces found for training")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            self.status_label.config(text="Status: Error training model")
    
    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.start_btn.config(text="Start Camera", bg="#4CAF50")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.status_label.config(text="Status: Camera stopped")
            self.found_label.config(text="")
        else:
            if not self.model_trained:
                messagebox.showwarning("Warning", "Please train the model first")
                return
                
            self.camera_active = True
            self.start_btn.config(text="Stop Camera", bg="#F44336")
            self.status_label.config(text="Status: Starting camera...")
            
            # Start camera in separate threads - one for capture, one for processing
            self.display_thread = threading.Thread(target=self.camera_loop)
            self.display_thread.daemon = True
            self.display_thread.start()
    
    def camera_loop(self):
        """Main camera loop - captures frames and updates display"""
        try:
            # Open camera with improved buffer settings
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution for speed
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                self.camera_active = False
                self.start_btn.config(text="Start Camera", bg="#4CAF50")
                self.status_label.config(text="Status: Camera error")
                return
            
            # Update status
            self.status_label.config(text="Status: Camera active")
            
            last_frame = None
            
            while self.camera_active:
                # Read a frame
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)  # Short delay to prevent CPU overload
                    continue
                
                # Make a copy for display
                display_frame = frame.copy()
                
                # Check if it's time for another detection
                current_time = time.time()
                if current_time - self.last_detection_time >= self.detection_interval:
                    self.last_detection_time = current_time
                    
                    # Start detection in a separate thread for each frame
                    detection_thread = threading.Thread(target=self.process_frame, args=(frame.copy(),))
                    detection_thread.daemon = True
                    detection_thread.start()
                
                # If we have results from a previous detection, display them
                if last_frame is not None:
                    display_frame = last_frame
                
                # Convert frame for display
                self.current_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Update the display
                self.update_frame()
                
                # Short sleep to prevent UI freezing
                time.sleep(0.01)
                
                # Get the latest frame with detections if available
                if self.frame_queue:
                    last_frame = self.frame_queue.pop()
                    self.frame_queue.clear()  # Clear any backlog
                
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")
            self.status_label.config(text="Status: Error")
        
        finally:
            if self.cap is not None:
                self.cap.release()
            self.camera_active = False
            self.start_btn.config(text="Start Camera", bg="#4CAF50")
    
    def process_frame(self, frame):
        """Process a single frame for face detection, recognition, and action detection"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optimize the image for faster detection
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with optimized parameters
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,  # Higher value for faster detection
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            threshold = 110
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Detect eyes within the face region
                eyes = self.eye_detector.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                
                
                # Determine action
                action = "Unknown"
                if len(eyes) == 0:  # No eyes detected
                    action = "Sleeping"
                elif len(eyes) == 1:  # One eye detected
                    action = "Winking"
                elif len(eyes) == 2:  # Two eyes detected
                    action = "Normal"
                    
                
                # Predict the face
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    
                    # Get name from label
                    if label in self.name_mapping:
                        name = self.name_mapping[label]
                        # Only show confident predictions
                        if confidence < threshold:  # Lower is better for LBPH
                            label_text = f"{name} ({confidence:.1f}) - {action}"
                            color = (0, 255, 0)  # Green for confident match
                            # Update UI with found name and action
                            self.found_label.config(text=f"Found: {name} | Action: {action}")
                        else:
                            label_text = f"Unknown - {action}"
                            color = (0, 0, 255)  # Red for unknown
                    else:
                        label_text = f"Unknown - {action}"
                        color = (0, 0, 255)  # Red
                except:
                    label_text = f"Error - {action}"
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add processed frame to queue for display
            self.frame_queue.append(frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def update_frame(self):
        """Update the UI with the current frame"""
        if self.current_frame is not None and self.camera_active:
            try:
                img = Image.fromarray(self.current_frame)
                # Resize image to fit the video frame
                img = self.resize_image(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            except Exception as e:
                print(f"Error updating frame: {e}")
    
    def resize_image(self, img):
        width = self.video_frame.winfo_width()
        height = self.video_frame.winfo_height()
        if width > 1 and height > 1:
            img_width, img_height = img.size
            # Calculate scaling factor
            scale = min(width/img_width, height/img_height)
            new_size = (int(img_width * scale), int(img_height * scale))
            return img.resize(new_size, Image.LANCZOS)
        return img
            
    def on_closing(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap is not None:
                self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()