import cv2
import numpy as np

class PerspectiveCorrector:
    def __init__(self):
        # Standard license plate dimensions (approximate ratios)
        self.standard_width = 400
        self.standard_height = 100
        
    def find_plate_corners(self, plate_image):
        """Find corners of the license plate for perspective correction"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (should be the plate boundary)
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # We need 4 corners for perspective correction
        if len(approx) == 4:
            return approx.reshape(4, 2)
        
        # If we don't get exactly 4 corners, use bounding rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return np.int0(box)
    
    def order_points(self, pts):
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        if pts is None or len(pts) != 4:
            return None
        
        # Sort by sum (top-left has smallest sum, bottom-right has largest)
        sum_pts = np.sum(pts, axis=1)
        top_left = pts[np.argmin(sum_pts)]
        bottom_right = pts[np.argmax(sum_pts)]
        
        # Sort by difference (top-right has smallest difference, bottom-left has largest)
        diff_pts = np.diff(pts, axis=1)
        top_right = pts[np.argmin(diff_pts)]
        bottom_left = pts[np.argmax(diff_pts)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    def correct_perspective(self, plate_image, corners=None):
        """Apply perspective correction to license plate"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        height, width = plate_image.shape[:2]
        
        # If corners are not provided, try to find them
        if corners is None:
            corners = self.find_plate_corners(plate_image)
        
        if corners is None:
            # If we can't find corners, return original image
            return plate_image
        
        # Order the corners
        ordered_corners = self.order_points(corners)
        if ordered_corners is None:
            return plate_image
        
        # Define destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [self.standard_width, 0],
            [self.standard_width, self.standard_height],
            [0, self.standard_height]
        ], dtype=np.float32)
        
        # Get perspective transform matrix
        try:
            matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
            
            # Apply perspective transformation
            corrected = cv2.warpPerspective(
                plate_image, matrix, (self.standard_width, self.standard_height)
            )
            
            return corrected
            
        except Exception as e:
            print(f"Error in perspective correction: {e}")
            return plate_image
    
    def auto_correct_perspective(self, plate_image):
        """Automatically detect and correct perspective distortion"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Try multiple methods for robustness
        methods = [
            self.correct_perspective_lines,
            self.correct_perspective_edges,
            self.correct_perspective_contours
        ]
        
        for method in methods:
            try:
                result = method(plate_image)
                if result is not None and result.size > 0:
                    return result
            except Exception as e:
                continue
        
        # If all methods fail, return original
        return plate_image
    
    def correct_perspective_lines(self, plate_image):
        """Correct perspective using line detection"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) < 2:
            return None
        
        # Find horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 20:  # Horizontal line
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < 20 or abs(angle + 90) < 20:  # Vertical line
                vertical_lines.append(line[0])
        
        # Need at least 2 horizontal and 2 vertical lines
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return None
        
        # Find intersections to get corners
        corners = self.find_line_intersections(horizontal_lines, vertical_lines)
        
        if corners is not None and len(corners) == 4:
            return self.correct_perspective(plate_image, corners)
        
        return None
    
    def correct_perspective_edges(self, plate_image):
        """Correct perspective using edge detection"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Apply morphological operations to enhance edges
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # Apply threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        corners = cv2.boxPoints(rect)
        
        return self.correct_perspective(plate_image, corners)
    
    def correct_perspective_contours(self, plate_image):
        """Correct perspective using contour analysis"""
        if plate_image is None or plate_image.size == 0:
            return None
        
        # This is a fallback method that applies basic skew correction
        # Convert to grayscale if needed
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()
        
        # Calculate skew angle
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) < 10:
            return plate_image
        
        # Fit line to find skew angle
        try:
            [vx, vy, x, y] = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.arctan2(vy, vx) * 180 / np.pi
            
            # Only correct if angle is significant
            if abs(angle) > 2:
                return self.rotate_image(plate_image, angle)
            
        except Exception as e:
            pass
        
        return plate_image
    
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        if image is None or image.size == 0:
            return None
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    
    def find_line_intersections(self, horizontal_lines, vertical_lines):
        """Find intersections between horizontal and vertical lines"""
        corners = []
        
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self.line_intersection(h_line, v_line)
                if intersection is not None:
                    corners.append(intersection)
        
        if len(corners) < 4:
            return None
        
        # Return the 4 corners that form the largest rectangle
        corners = np.array(corners)
        hull = cv2.convexHull(corners)
        
        if len(hull) >= 4:
            return hull[:4].reshape(4, 2)
        
        return None
    
    def line_intersection(self, line1, line2):
        """Find intersection point between two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))
    
    def enhance_corrected_image(self, corrected_image):
        """Enhance the corrected image for better OCR"""
        if corrected_image is None or corrected_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(corrected_image.shape) == 3:
            gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = corrected_image.copy()
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        return sharpened 
