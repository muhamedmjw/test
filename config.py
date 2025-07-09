 # Configuration settings for License Plate Recognition System

# Database Configuration
DATABASE_PATH = "plates.db"

# Camera Configuration
# CHANGE THIS: Set to 0 for default webcam, 1 for external camera, or IP camera URL
CAMERA_SOURCE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# OCR Configuration
OCR_LANGUAGES = ['en']  # English for Iraqi plates
OCR_GPU = False  # Set to True if you have CUDA GPU for faster processing

# Iraqi License Plate Configuration
VALID_CITY_CODES = ['20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50']
VALID_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
PLATE_PATTERN = r'^(\d{2})\s*([A-Z])\s*(\d{5})$'

# GUI Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FONT_SIZE = 12

# Image Processing Configuration
MIN_PLATE_AREA = 500  # Minimum area for plate detection
MAX_PLATE_AREA = 50000  # Maximum area for plate detection
PLATE_ASPECT_RATIO_MIN = 2.0  # Minimum width/height ratio for plates
PLATE_ASPECT_RATIO_MAX = 6.0  # Maximum width/height ratio for plates

# Gate Control Configuration
GATE_OPEN_DURATION = 3  # seconds to keep gate open
AUTO_CLOSE_GATE = True