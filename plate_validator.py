import re
from config import VALID_CITY_CODES, VALID_LETTERS, PLATE_PATTERN

class PlateValidator:
    def __init__(self):
        self.pattern = re.compile(PLATE_PATTERN)
    
    def validate_and_parse(self, text):
        """
        Validate and parse Iraqi license plate text
        Returns: (is_valid, city_code, letter, number) or (False, None, None, None)
        """
        if not text:
            return False, None, None, None
        
        # Clean the text - remove extra spaces, convert to uppercase
        cleaned_text = ' '.join(text.upper().split())
        
        # Try to match the pattern
        match = self.pattern.match(cleaned_text)
        
        if not match:
            return False, None, None, None
        
        city_code, letter, number = match.groups()
        
        # Validate city code
        if city_code not in VALID_CITY_CODES:
            return False, None, None, None
        
        # Validate letter
        if letter not in VALID_LETTERS:
            return False, None, None, None
        
        # Validate number (should be exactly 5 digits)
        if len(number) != 5 or not number.isdigit():
            return False, None, None, None
        
        return True, city_code, letter, number
    
    def format_plate(self, city_code, letter, number):
        """Format plate components into standard format"""
        return f"{city_code} {letter} {number}"
    
    def preprocess_ocr_text(self, ocr_results):
        """
        Preprocess OCR results to improve plate detection
        OCR results format: [(text, confidence, bbox), ...]
        """
        valid_plates = []
        
        for text, confidence, bbox in ocr_results:
            # Skip low confidence results
            if confidence < 0.5:
                continue
            
            # Clean and preprocess text
            cleaned_text = self.clean_ocr_text(text)
            
            # Try to validate
            is_valid, city_code, letter, number = self.validate_and_parse(cleaned_text)
            
            if is_valid:
                valid_plates.append({
                    'city_code': city_code,
                    'letter': letter,
                    'number': number,
                    'full_plate': self.format_plate(city_code, letter, number),
                    'confidence': confidence,
                    'bbox': bbox,
                    'original_text': text
                })
        
        # Sort by confidence (highest first)
        valid_plates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return valid_plates
    
    def clean_ocr_text(self, text):
        """Clean OCR text to improve recognition"""
        if not text:
            return ""
        
        # Common OCR mistakes and corrections
        corrections = {
            'O': '0',  # O to 0
            'o': '0',  # o to 0
            'I': '1',  # I to 1
            'l': '1',  # l to 1
            'S': '5',  # S to 5 (sometimes)
            'G': '6',  # G to 6 (sometimes)
            'B': '8',  # B to 8 (sometimes)
        }
        
        # Apply corrections
        cleaned = text.upper()
        
        # Remove non-alphanumeric characters except spaces
        cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
        
        # Try to fix common OCR mistakes in numbers
        # This is a simple approach - you might need to refine based on actual OCR results
        parts = cleaned.split()
        if len(parts) >= 3:
            # Assuming format: citycode letter number
            city_part = parts[0]
            letter_part = parts[1]
            number_part = ''.join(parts[2:])  # Join remaining parts
            
            # Clean city code (should be 2 digits)
            city_part = ''.join(c if c.isdigit() else corrections.get(c, c) for c in city_part)
            
            # Clean letter (should be 1 letter)
            letter_part = ''.join(c if c.isalpha() else '' for c in letter_part)[:1]
            
            # Clean number (should be 5 digits)
            number_part = ''.join(c if c.isdigit() else corrections.get(c, c) for c in number_part)
            
            cleaned = f"{city_part} {letter_part} {number_part}"
        
        return cleaned.strip()
    
    def suggest_corrections(self, text):
        """
        Suggest possible corrections for invalid plates
        Useful for manual verification
        """
        suggestions = []
        
        if not text:
            return suggestions
        
        cleaned = self.clean_ocr_text(text)
        parts = cleaned.split()
        
        if len(parts) >= 3:
            city_part = parts[0]
            letter_part = parts[1]
            number_part = ''.join(parts[2:])
            
            # Try different city codes if current one is invalid
            if city_part not in VALID_CITY_CODES:
                for valid_city in VALID_CITY_CODES:
                    if len(city_part) == 2 and self.similar_digits(city_part, valid_city):
                        suggestion = f"{valid_city} {letter_part} {number_part}"
                        is_valid, _, _, _ = self.validate_and_parse(suggestion)
                        if is_valid:
                            suggestions.append(suggestion)
            
            # Try different letters if current one is invalid
            if letter_part not in VALID_LETTERS:
                for valid_letter in VALID_LETTERS:
                    if self.similar_chars(letter_part, valid_letter):
                        suggestion = f"{city_part} {valid_letter} {number_part}"
                        is_valid, _, _, _ = self.validate_and_parse(suggestion)
                        if is_valid:
                            suggestions.append(suggestion)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def similar_digits(self, str1, str2):
        """Check if two strings are similar (for digit corrections)"""
        if len(str1) != len(str2):
            return False
        
        differences = sum(c1 != c2 for c1, c2 in zip(str1, str2))
        return differences <= 1  # Allow 1 character difference
    
    def similar_chars(self, char1, char2):
        """Check if two characters are similar (common OCR mistakes)"""
        similar_pairs = [
            ('O', '0'), ('I', '1'), ('S', '5'), ('G', '6'), ('B', '8'),
            ('Z', '2'), ('A', '4'), ('D', '0'), ('U', 'V'), ('H', 'N')
        ]
        
        for pair in similar_pairs:
            if (char1 in pair and char2 in pair) or char1 == char2:
                return True
        
        return False 
