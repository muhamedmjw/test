import sqlite3
import os
from datetime import datetime
from config import DATABASE_PATH

class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create license_plates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS license_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city_code TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    number TEXT NOT NULL,
                    full_plate TEXT NOT NULL UNIQUE,
                    vehicle_status TEXT DEFAULT 'OUT',
                    is_authorized INTEGER DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    entry_count INTEGER DEFAULT 0,
                    notes TEXT
                )
            ''')
            
            # Create activity_log table for tracking all activities
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_id INTEGER,
                    full_plate TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_manual INTEGER DEFAULT 0,
                    notes TEXT,
                    FOREIGN KEY (plate_id) REFERENCES license_plates (id)
                )
            ''')
            
            conn.commit()
    
    def add_or_update_plate(self, city_code, letter, number, is_authorized=0):
        """Add a new plate or update existing one"""
        full_plate = f"{city_code} {letter} {number}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if plate exists
            cursor.execute("SELECT id FROM license_plates WHERE full_plate = ?", (full_plate,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing plate
                cursor.execute('''
                    UPDATE license_plates 
                    SET is_authorized = ?, last_seen = CURRENT_TIMESTAMP
                    WHERE full_plate = ?
                ''', (is_authorized, full_plate))
                return existing[0]
            else:
                # Add new plate
                cursor.execute('''
                    INSERT INTO license_plates 
                    (city_code, letter, number, full_plate, is_authorized, entry_count)
                    VALUES (?, ?, ?, ?, ?, 0)
                ''', (city_code, letter, number, full_plate, is_authorized))
                return cursor.lastrowid
    
    def get_plate_info(self, full_plate):
        """Get complete information about a plate"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, city_code, letter, number, full_plate, vehicle_status, 
                       is_authorized, first_seen, last_seen, entry_count, notes
                FROM license_plates WHERE full_plate = ?
            ''', (full_plate,))
            return cursor.fetchone()
    
    def process_plate_entry(self, city_code, letter, number, is_manual=False, manual_action=None):
        """Process a plate entry and update vehicle status"""
        full_plate = f"{city_code} {letter} {number}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get current plate info
            plate_info = self.get_plate_info(full_plate)
            
            if plate_info:
                plate_id, _, _, _, _, current_status, is_authorized, _, _, entry_count, _ = plate_info
                
                # Handle manual override
                if is_manual and manual_action:
                    if manual_action == "GRANT":
                        new_status = 'IN'
                        action = 'MANUAL_GRANT'
                    elif manual_action == "DENY":
                        new_status = current_status  # Keep current status
                        action = 'MANUAL_DENY'
                    elif manual_action == "RESET":
                        new_status = 'OUT'
                        action = 'MANUAL_RESET'
                    else:
                        new_status = 'IN' if current_status == 'OUT' else 'OUT'
                        action = 'MANUAL_TOGGLE'
                else:
                    # Automatic processing
                    if is_authorized:
                        # Toggle status for authorized vehicles
                        new_status = 'IN' if current_status == 'OUT' else 'OUT'
                        action = 'ENTER' if new_status == 'IN' else 'EXIT'
                        
                        # Increment entry count only when entering
                        if new_status == 'IN':
                            entry_count += 1
                    else:
                        # Unauthorized vehicle - don't change status
                        new_status = current_status
                        action = 'DENIED'
                
                # Update plate info
                cursor.execute('''
                    UPDATE license_plates 
                    SET vehicle_status = ?, last_seen = CURRENT_TIMESTAMP, entry_count = ?
                    WHERE id = ?
                ''', (new_status, entry_count, plate_id))
                
            else:
                # New plate
                plate_id = self.add_or_update_plate(city_code, letter, number, 0)
                if is_manual and manual_action == "GRANT":
                    new_status = 'IN'
                    action = 'MANUAL_GRANT'
                else:
                    new_status = 'OUT'
                    action = 'DENIED'
            
            # Log the activity
            self.log_activity(plate_id, full_plate, action, is_manual)
            
            # Return updated plate info
            return self.get_plate_info(full_plate)
    
    def log_activity(self, plate_id, full_plate, action, is_manual=False, notes=None):
        """Log an activity to the activity log"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO activity_log (plate_id, full_plate, action, is_manual, notes)
                VALUES (?, ?, ?, ?, ?)
            ''', (plate_id, full_plate, action, 1 if is_manual else 0, notes))
    
    def get_recent_activity(self, limit=10):
        """Get recent activity log entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT full_plate, action, timestamp, is_manual, notes
                FROM activity_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
    
    def get_all_authorized_plates(self):
        """Get all authorized plates"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT full_plate, vehicle_status, last_seen, entry_count
                FROM license_plates 
                WHERE is_authorized = 1
                ORDER BY last_seen DESC
            ''')
            return cursor.fetchall()
    
    def set_plate_authorization(self, full_plate, is_authorized):
        """Set authorization status for a plate"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE license_plates 
                SET is_authorized = ?, last_seen = CURRENT_TIMESTAMP
                WHERE full_plate = ?
            ''', (1 if is_authorized else 0, full_plate))
            
            # Log the action
            action = 'AUTHORIZED' if is_authorized else 'UNAUTHORIZED'
            self.log_activity(None, full_plate, action, True)
    
    def get_plate_statistics(self):
        """Get basic statistics about plates"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total plates
            cursor.execute("SELECT COUNT(*) FROM license_plates")
            total_plates = cursor.fetchone()[0]
            
            # Authorized plates
            cursor.execute("SELECT COUNT(*) FROM license_plates WHERE is_authorized = 1")
            authorized_plates = cursor.fetchone()[0]
            
            # Vehicles currently inside
            cursor.execute("SELECT COUNT(*) FROM license_plates WHERE vehicle_status = 'IN'")
            inside_vehicles = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM activity_log 
                WHERE timestamp >= datetime('now', '-1 day')
            ''')
            recent_activity = cursor.fetchone()[0]
            
            return {
                'total_plates': total_plates,
                'authorized_plates': authorized_plates,
                'inside_vehicles': inside_vehicles,
                'recent_activity': recent_activity
            }
