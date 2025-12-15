"""
Handwritten Digit Recognizer using MNIST and Neural Networks
This module provides real-time handwritten digit recognition using a webcam feed.
It trains a Multi-Layer Perceptron (MLP) classifier on the MNIST dataset in the 
background and performs live predictions on digits drawn in a centered ROI box.
Classes:
    DigitRecognizer: Handles model training and digit prediction with preprocessing.
Main Features:
    - Background training thread to avoid blocking the UI
    - Adaptive image preprocessing to match MNIST format (28x28 grayscale)
    - Real-time webcam capture with visual feedback
    - Confidence scoring for predictions
    - Debug visualization showing the neural network's input view
Requirements (pip install):
    - opencv-python (cv2)
    - numpy
    - scikit-learn (sklearn)
    - scipy (dependency of scikit-learn)
Usage:
    python digit_class.py
    Press 'q' to quit the application.
    Draw digits in the green bounding box for real-time recognition.
"""

import cv2
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import threading
import time

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.is_ready = False
        self.training_status = "Initializing..."
        self.prediction = "..."
        self.confidence = 0.0

    def train_background(self):
        """Downloads MNIST data and trains a lightweight Neural Net in the background."""
        def _train():
            self.training_status = "Downloading MNIST (may take a min)..."
            print("--- [System] Downloading MNIST dataset... ---")
            
            # Fetch data (28x28 pixel images = 784 features)
            # We use a subset (15k samples) to keep startup fast while maintaining accuracy
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
            X = X[:15000] / 255.0  # Normalize pixel values (0-1)
            y = y[:15000]

            self.training_status = "Training Neural Network..."
            print("--- [System] Training Neural Network... ---")
            
            # Simple Multi-Layer Perceptron
            # Hidden layers: 100 neurons. Increased max_iter to prevent ConvergenceWarning
            self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=50, alpha=1e-4,
                                       solver='sgd', verbose=False, random_state=1,
                                       learning_rate_init=0.1)
            self.model.fit(X, y)
            
            self.is_ready = True
            self.training_status = "Active"
            print("--- [System] Model Ready! ---")

        thread = threading.Thread(target=_train)
        thread.daemon = True
        thread.start()

    def predict(self, roi_img):
        if not self.is_ready:
            return
        
        # 1. Image Pre-processing pipeline to match MNIST format
        # ROI is expected to be Grayscale already
        
        # Threshold to separate ink from paper
        # Adaptive thresholding handles different lighting conditions
        thresh = cv2.adaptiveThreshold(roi_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # 2. Centering Logic (Critical for MNIST accuracy)
        # Find the bounding box of the digit so we don't pass empty whitespace
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (the digit)
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                
                # Crop to the digit
                digit_roi = thresh[y:y+h, x:x+w]
                
                # Resize to 20x20 keeping aspect ratio within a 28x28 box (MNIST style)
                # Create a black square canvas
                canvas = np.zeros((28, 28), dtype=np.uint8)
                
                # Scale logic
                scale = 20.0 / max(w, h)
                
                # FIX: Ensure dimensions are at least 1 pixel to prevent OpenCV crash on thin digits
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                
                resized_digit = cv2.resize(digit_roi, (nw, nh))
                
                # Center it on the 28x28 canvas
                dx = (28 - nw) // 2
                dy = (28 - nh) // 2
                canvas[dy:dy+nh, dx:dx+nw] = resized_digit
                
                # Flatten and Normalize
                final_input = canvas.reshape(1, -1) / 255.0
                
                # Predict
                probs = self.model.predict_proba(final_input)[0]
                prediction = np.argmax(probs)
                self.confidence = probs[prediction]
                
                if self.confidence > 0.5: # Only show if somewhat confident
                    self.prediction = str(prediction)
                else:
                    self.prediction = "?"
                
                return canvas # Return processed image for debug view
        
        return thresh # Fallback

def main():
    # Initialize Recognizer
    recognizer = DigitRecognizer()
    recognizer.train_background()

    cap = cv2.VideoCapture(0)
    
    # UI Constants
    BOX_SIZE = 200
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Define Region of Interest (ROI) Box in center
        box_x = (w - BOX_SIZE) // 2
        box_y = (h - BOX_SIZE) // 2
        
        # Extract and Process ROI
        roi = frame[box_y:box_y+BOX_SIZE, box_x:box_x+BOX_SIZE]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0) # Reduce noise
        
        # Run Prediction
        debug_view = recognizer.predict(gray_roi)

        # --- DRAWING GUI ---
        
        # 1. Main Bounding Box (Green)
        color = (0, 255, 0) if recognizer.is_ready else (0, 165, 255)
        cv2.rectangle(frame, (box_x, box_y), (box_x+BOX_SIZE, box_y+BOX_SIZE), color, 2)
        
        # 2. Status Text
        cv2.putText(frame, f"System: {recognizer.training_status}", (10, 30), font, 0.7, (200, 200, 200), 2)
        
        # 3. Prediction Display
        if recognizer.is_ready:
            # Show the prediction large
            cv2.putText(frame, f"Digit: {recognizer.prediction}", (10, 100), font, 2, (0, 255, 0), 3)
            cv2.putText(frame, f"Conf: {recognizer.confidence:.2f}", (10, 140), font, 0.6, (0, 255, 0), 1)
            
            # Show the "Neural View" (what the computer actually sees)
            if debug_view is not None:
                # Resize the tiny 28x28 debug image up to 100x100 to show user
                debug_upscaled = cv2.resize(debug_view, (100, 100), interpolation=cv2.INTER_NEAREST)
                debug_upscaled = cv2.cvtColor(debug_upscaled, cv2.COLOR_GRAY2BGR)
                
                # Overlay in corner
                frame[h-120:h-20, w-120:w-20] = debug_upscaled
                cv2.putText(frame, "Neural View", (w-120, h-125), font, 0.5, (255, 255, 255), 1)
                cv2.rectangle(frame, (w-122, h-122), (w-18, h-18), (255, 255, 0), 1)

        cv2.imshow('Handwritten Digit Recognizer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()