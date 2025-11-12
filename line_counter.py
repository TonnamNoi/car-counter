"""
LineCounter Module - Traffic Car Counting System
\nThis is the core counting logic
"""

import time
from typing import Tuple, Dict


class Geometry:
    #Handles geometric calculations for line crossing detection
    
    @staticmethod
    def line_side(point: Tuple[float, float], 
                line_start: Tuple[float, float], 
                line_end: Tuple[float, float]) -> float:
        """
        Determine which side of the line a point is on using cross product
        Returns: positive = one side, negative = other side
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    @staticmethod
    def crossed(previous_side: float, current_side: float) -> bool:
        #Check if point crossed the line (sides have different signs)
        return previous_side * current_side < 0
    
    @staticmethod
    def direction_sign(previous_side: float, current_side: float) -> int:
        """
        Determine crossing direction
        Returns: +1 for one direction, -1 for opposite, 0 for no crossing
        """
        if previous_side > 0 and current_side < 0:
            return 1
        elif previous_side < 0 and current_side > 0:
            return -1
        return 0
    
    @staticmethod
    def centroid_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate center point of bounding box
        bbox format: (x1, y1, x2, y2)
        Returns: (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy


class LineCounter:
    """
    Main car counting system
    Counts vehicles crossing a virtual line with direction tracking
    """
    
    def __init__(self, 
                line_start: Tuple[float, float],
                line_end: Tuple[float, float],
                cooldown: float = 1.0):
        """
        Initialize the LineCounter
        
        Args:
            line_start: (x, y) start point of counting line
            line_end: (x, y) end point of counting line
            cooldown: seconds to wait before counting same car again
        """
        self.line_start = line_start
        self.line_end = line_end
        self.cooldown = cooldown
        
        # Count variables
        self.count_in = 0
        self.count_out = 0
        
        # Track each car state
        self.tracked_objects: Dict[int, Dict] = {}
        
        # Geometry helper
        self.geometry = Geometry()
        
        # Statistics
        self.total_crossed = 0
        self.start_time = time.time()
    
    def update(self, 
            track_id: int, 
            centroid: Tuple[float, float], 
            timestamp: float) -> bool:
        """
        Update counter with new car position
        
        Args:
            track_id: Unique ID for the car
            centroid: (x, y) center position of car
            timestamp: Current time in seconds
            
        Returns:
            bool: True if this car just crossed the line
        """
        # Calculate which side of line the car is on
        current_side = self.geometry.line_side(centroid, self.line_start, self.line_end)
        
        # First time seeing this car
        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = {
                'side': current_side,
                'last_time': timestamp
            }
            return False
        
        # Get previous state
        previous_state = self.tracked_objects[track_id]
        previous_side = previous_state['side']
        last_time = previous_state['last_time']
        
        # Check cooldown
        time_since_last = timestamp - last_time
        if time_since_last < self.cooldown:
            self.tracked_objects[track_id]['side'] = current_side
            return False
        
        # Check if car crossed the line
        if self.geometry.crossed(previous_side, current_side):
            direction = self.geometry.direction_sign(previous_side, current_side)
            
            if direction > 0:
                self.count_in += 1
            elif direction < 0:
                self.count_out += 1
            
            self.total_crossed += 1
            
            # Update tracking state
            self.tracked_objects[track_id] = {
                'side': current_side,
                'last_time': timestamp
            }
            
            return True
        
        # No crossing - just update position
        self.tracked_objects[track_id]['side'] = current_side
        return False
    
    @property
    def total_count(self) -> int:
        # Get total number of cars counted
        return self.count_in + self.count_out
    
    def get_line_coordinates(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # Get the counting line coordinates for drawing
        return self.line_start, self.line_end
    
    def get_statistics(self) -> Dict:
        # Get all counting statistics
        elapsed_time = time.time() - self.start_time
        
        return {
            'count_in': self.count_in,
            'count_out': self.count_out,
            'total': self.total_count,
            'elapsed_time': elapsed_time,
            'cars_per_minute': (self.total_count / elapsed_time * 60) if elapsed_time > 0 else 0,
            'active_tracks': len(self.tracked_objects)
        }
    
    def reset(self):
        # Reset all counts to zero
        self.count_in = 0
        self.count_out = 0
        self.total_crossed = 0
        self.tracked_objects.clear()
        self.start_time = time.time()
    
    def cleanup_old_tracks(self, current_time: float, max_age: float = 5.0):
        # Remove tracks that haven't been updated recently
        tracks_to_remove = []
        
        for track_id, state in self.tracked_objects.items():
            age = current_time - state['last_time']
            if age > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]