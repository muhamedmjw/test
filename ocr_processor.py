import cv2
import numpy as np
import easyocr
from config import OCR_LANGUAGES, OCR_GPU

class OCRProcessor:
    def __init__(self):
        self.reader = None
        self.init_ocr()
    
    def init_ocr(self):
        """Initialize EasyOCR reader"""
        try:
            self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)
            print("OCR reader initialized successfully")
        except Exception as e:
            print(f"Error initializing OCR: {e}")
            self.reader = None
    
    def extract_text_from_image(self, image):
        """Extract text from image using OCR"""
        if self.reader is None:
            print("OCR reader not initialized")
            return []
        
        if image is None or image.size == 0:
            return []
        
        try:
            # EasyOCR works with both color and grayscale images
            results = self.reader.readtext(image)
            
            # Format results: [(text, confidence, bbox), ...]
            formatted_results = []
            for (bbox, text, confidence) in results:
                formatted_results.append((text, confidence, bbox))
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            return []
    
    def extract_text_from_regions(self, plate_regions):
        """Extract text from multiple plate regions"""
        all_results = []
        
        for region in plate_regions:
            # Try with original image first
            results = self.extract_text_from_image(region['image'])
            
            # If no good results, try with processed image
            if not results and 'processed_image' in region:
                results = self.extract_text_from_image(region['processed_image'])
            
            # Add region info to results
            for text, confidence, bbox in results:
                all_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'region_bbox': region['bbox'],
                    'method': region.get('method', 'unknown')
                })
        
        return all_results
    
    def preprocess_for_ocr(self, image):
        """Preprocess image specifically for OCR"""
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if too small (OCR works better with larger images)
        height, width = gray.shape
        if height < 50 or width < 150:
            scale_factor = max(50 / height, 150 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=0)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def extract_with_multiple_methods(self, image):
        """Try multiple preprocessing methods for better OCR results"""
        if image is None or image.size == 0:
            return []
        
        all_results = []
        
        # Method 1: Original image
        results1 = self.extract_text_from_image(image)
        all_results.extend(results1)
        
        # Method 2: Preprocessed image
        preprocessed = self.preprocess_for_ocr(image)
        if preprocessed is not None:
            results2 = self.extract_text_from_image(preprocessed)
            all_results.extend(results2)
        
        # Method 3: Inverted image (white text on black background)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        inverted = cv2.bitwise_not(gray)
        results3 = self.extract_text_from_image(inverted)
        all_results.extend(results3)
        
        # Method 4: Thresholded image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results4 = self.extract_text_from_image(thresh)
        all_results.extend(results4)
        
        # Remove duplicates and sort by confidence
        unique_results = self.remove_duplicate_results(all_results)
        unique_results.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
        
        return unique_results
    
    def remove_duplicate_results(self, results):
        """Remove duplicate OCR results"""
        if not results:
            return []
        
        unique_results = []
        seen_texts = set()
        
        for text, confidence, bbox in results:
            # Normalize text for comparison
            normalized_text = ''.join(text.upper().split())
            
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_results.append((text, confidence, bbox))
        
        return unique_results
    
    def enhance_image_for_ocr(self, image):
        """Enhanced preprocessing specifically for license plate OCR"""
        if image is None or image.size == 0:
            return None
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth
        blurred = cv2.GaussianBlur(morph, (3, 3), 0)
        
        return blurred
    
    def process_plate_image(self, plate_image):
        """Main method to process a single plate image"""
        if plate_image is None or plate_image.size == 0:
            return []
        
        # Try multiple enhancement methods
        methods = [
            self.extract_text_from_image,
            lambda img: self.extract_text_from_image(self.preprocess_for_ocr(img)),
            lambda img: self.extract_text_from_image(self.enhance_image_for_ocr(img))
        ]
        
        all_results = []
        
        for method in methods:
            try:
                results = method(plate_image)
                all_results.extend(results)
            except Exception as e:
                print(f"Error in OCR method: {e}")
                continue
        
        # Remove duplicates and return best results
        unique_results = self.remove_duplicate_results(all_results)
        
        # Filter by confidence threshold
        filtered_results = [(text, conf, bbox) for text, conf, bbox in unique_results if conf > 0.3]
        
        # Sort by confidence
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_results
    
    def get_best_text_result(self, results):
        """Get the best text result from OCR results"""
        if not results:
            return None
        
        # Filter by minimum confidence
        good_results = [r for r in results if r[1] > 0.5]
        
        if not good_results:
            # If no good results, return the best available
            return results[0] if results else None
        
        # Return the highest confidence result
        return good_results[0]
    
    def batch_process_regions(self, plate_regions):
        """Process multiple plate regions efficiently"""
        results = []
        
        for i, region in enumerate(plate_regions):
            print(f"Processing region {i+1}/{len(plate_regions)}")
            
            # Process the region
            ocr_results = self.process_plate_image(region['image'])
            
            # Add region information
            for text, confidence, bbox in ocr_results:
                results.append({
                    'region_id': i,
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'region_bbox': region['bbox'],
                    'method': region.get('method', 'unknown')
                })
        
        return results 
