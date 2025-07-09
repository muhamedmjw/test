import cv2
import numpy as np
from config import MIN_PLATE_AREA, MAX_PLATE_AREA, PLATE_ASPECT_RATIO_MIN, PLATE_ASPECT_RATIO_MAX

class PlateDetector:
    def __init__(self):
        # Initialize cascade classifier for plate detection
        # You can download this from: https://github.com/opencv/opencv/tree/master/data/haarcascades
        # CHANGE THIS: Download and place the cascade file in your project directory
        self.cascade_path = "haarcascade_license_plate_rus_16stages.xml"
        self.plate_cascade = None
        self.load_cascade()
    
    def load_cascade(self):
        """Load Haar cascade for plate detection"""
        try:
            self.plate_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.plate_cascade.empty():
                print("Warning: Could not load cascade classifier")
                self.plate_cascade = None
        except Exception as e:
            print(f"Error loading cascade: {e}")
            self.plate_cascade = None
    
    def detect_plates_cascade(self, frame):
        """Detect plates using Haar cascade"""
        if self.plate_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect plates
        plates = self.plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return plates
    
    def detect_plates_contour(self, frame):
        """Detect plates using contour detection method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges using Canny
        edges = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_plates = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < MIN_PLATE_AREA or area > MAX_PLATE_AREA:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = w / h
            
            # Filter by aspect ratio (license plates are typically wider than tall)
            if aspect_ratio < PLATE_ASPECT_RATIO_MIN or aspect_ratio > PLATE_ASPECT_RATIO_MAX:
                continue
            
            # Calculate extent (contour area / bounding rectangle area)
            rect_area = w * h
            extent = area / rect_area
            
            # Filter by extent (should be reasonable for a filled rectangle)
            if extent < 0.3:
                continue
            
            potential_plates.append((x, y, w, h, area, aspect_ratio))
        
        # Sort by area (larger plates first)
        potential_plates.sort(key=lambda x: x[4], reverse=True)
        
        return potential_plates
    
    def detect_plates_combined(self, frame):
        """Combine both detection methods for better results"""
        detections = []
        
        # Method 1: Cascade detection
        cascade_plates = self.detect_plates_cascade(frame)
        for (x, y, w, h) in cascade_plates:
            detections.append({
                'bbox': (x, y, w, h),
                'area': w * h,
                'aspect_ratio': w / h,
                'method': 'cascade',
                'confidence': 0.8  # Assume good confidence for cascade
            })
        
        # Method 2: Contour detection
        contour_plates = self.detect_plates_contour(frame)
        for (x, y, w, h, area, aspect_ratio) in contour_plates:
            detections.append({
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'method': 'contour',
                'confidence': 0.6  # Lower confidence for contour method
            })
        
        # Remove duplicates (overlapping detections)
        filtered_detections = self.remove_overlapping_detections(detections)
        
        # Sort by confidence and area
        filtered_detections.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
        
        return filtered_detections
    
    def remove_overlapping_detections(self, detections):
        """Remove overlapping bounding boxes"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence first
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            x1, y1, w1, h1 = detection['bbox']
            
            # Check if this detection overlaps significantly with any already filtered detection
            is_duplicate = False
            for filtered_detection in filtered:
                x2, y2, w2, h2 = filtered_detection['bbox']
                
                # Calculate overlap
                overlap_area = self.calculate_overlap(x1, y1, w1, h1, x2, y2, w2, h2)
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If overlap is more than 50% of either area, consider it duplicate
                if overlap_area > 0.5 * min(area1, area2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def calculate_overlap(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """Calculate overlap area between two rectangles"""
        # Calculate intersection coordinates
        left = max(x1, x2)
        right = min(x1 + w1, x2 + w2)
        top = max(y1, y2)
        bottom = min(y1 + h1, y2 + h2)
        
        # Check if there's an intersection
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        return 0
    
    def extract_plate_regions(self, frame, detections):
        """Extract plate regions from frame"""
        plate_regions = []
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            
            # Add some padding around the detected region
            padding = 5
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(frame.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(frame.shape[0] - y_padded, h + 2 * padding)
            
            # Extract region
            plate_region = frame[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
            
            if plate_region.size > 0:
                plate_regions.append({
                    'image': plate_region,
                    'bbox': (x_padded, y_padded, w_padded, h_padded),
                    'original_bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'method': detection['method']
                })
        
        return plate_regions
    
    def preprocess_plate_region(self, plate_image):
        """Preprocess extracted plate region for better OCR"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Resize if too small
        height, width = gray.shape
        if height < 40 or width < 100:
            scale_factor = max(40 / height, 100 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def draw_detections(self, frame, detections):
        """Draw detection bounding boxes on frame"""
        result_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            confidence = detection['confidence']
            method = detection['method']
            
            # Choose color based on method
            color = (0, 255, 0) if method == 'cascade' else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence text
            text = f"{method}: {confidence:.2f}"
            cv2.putText(result_frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result_frame
    
    def detect_and_extract_plates(self, frame):
        """Main method to detect and extract plates from frame"""
        if frame is None:
            return [], []
        
        # Detect plates
        detections = self.detect_plates_combined(frame)
        
        # Extract plate regions
        plate_regions = self.extract_plate_regions(frame, detections)
        
        # Preprocess plate regions
        processed_regions = []
        for region in plate_regions:
            processed_image = self.preprocess_plate_region(region['image'])
            if processed_image is not None:
                region['processed_image'] = processed_image
                processed_regions.append(region)
        
        return detections, processed_regions