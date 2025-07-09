import cv2
import numpy as np
from config import CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_connected = False
        self.current_frame = None
        self.init_camera()
    
    def init_camera(self):
        """Initialize camera connection"""
        try:
            # CHANGE THIS: Modify CAMERA_SOURCE in config.py to match your camera
            self.cap = cv2.VideoCapture(CAMERA_SOURCE)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {CAMERA_SOURCE}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_connected = True
            print(f"Camera initialized successfully: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.is_connected = False
            return False
    
    def get_frame(self):
        """Get current frame from camera"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                return frame
            else:
                print("Failed to capture frame")
                return None
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def get_processed_frame(self):
        """Get frame with basic preprocessing for plate detection"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Apply basic preprocessing
        processed = self.preprocess_frame(frame)
        return processed
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve plate detection"""
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def enhance_frame_quality(self, frame):
        """Enhance frame quality for better OCR results"""
        if frame is None:
            return None
        
        # Increase contrast and brightness
        enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        
        # Apply histogram equalization for better contrast
        if len(enhanced.shape) == 3:
            # For color images
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
            enhanced[:, :, 0] = cv2.equalizeHist(enhanced[:, :, 0])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YUV2BGR)
        else:
            # For grayscale images
            enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
    def get_frame_for_gui(self):
        """Get frame formatted for GUI display"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        # Resize if needed for GUI display
        height, width = frame.shape[:2]
        
        # Scale down if too large
        max_width = 400
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def save_frame(self, filename=None):
        """Save current frame to file"""
        if self.current_frame is None:
            return False
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_frame_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, self.current_frame)
            print(f"Frame saved as {filename}")
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
    
    def get_camera_info(self):
        """Get camera information"""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'source': CAMERA_SOURCE
            }
        except Exception as e:
            print(f"Error getting camera info: {e}")
            return None
    
    def reconnect(self):
        """Attempt to reconnect camera"""
        self.close()
        return self.init_camera()
    
    def close(self):
        """Close camera connection"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        print("Camera connection closed")
    
    def __del__(self):
        """Destructor to ensure camera is closed"""
        self.close() 
