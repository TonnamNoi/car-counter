"""
Car Counter System
Integrates YOLO detection + ByteTrack + LineCounter
Processes video and creates output with car counting visualization
"""

# pip install ultralytics opencv-python numpy

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
from line_counter import LineCounter, Geometry

print("="*70)
print("CAR COUNTER SYSTEM")
print("="*70)

# ============================================================================
# CONFIGURATION - Adjust these settings
# ============================================================================

INPUT_VIDEO = "traffic.mp4"  # input video file
OUTPUT_DIR = "output"        # where to save results
OUTPUT_VIDEO = "counted_video.mp4"

"""
Line position (adjust based on video)
Format: (x1, y1, x2, y2) - draws a line from (x1,y1) to (x2,y2)
These are percentage-based, will be scaled to actual video size
"""
# Uncomment to change line position
LINE_POSITION = (0.1, 0.6, 0.9, 0.6)  # Horizontal line | 60% height
#LINE_POSITION = (0.5, 0.1, 0.5, 0.9)  # Vertical line | 50% screen width (center)

#LINE_POSITION = (0.0, 0.4, 1.9, 0.4) # In case of vertical video, custom line (diagonal?)

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # Autoloaded model file when run
# test using yolov8n.pt model (small, fast, stable)
# lastest yolo11n.pt (~3% more accurate, ~10% faster processing, might have undiscovered bug)
CONFIDENCE_THRESHOLD = 0.3 # Detection confidence range(0.0 to 1.0) # use 0.3 when video test

# Detection confidence Selection Rules
"""
Set it Too LOW (0.0 or 0.1)
Pros: won't miss real detection, very sensitive
Cons: tons of false positive (detect things that aren't there)
Example: label Shadow as a Car, group 2 car as 1 box

Set it Too HIGH (0.9 or 1.0)
Pros: few false positive
Cons: will miss many real detection (become overly strict)
Example: ignore small or blurry cars bc it's not 95% sure
"""

# Display settings
SHOW_LIVE_PREVIEW = False  # Set True to see processing in real-time (slower)
SKIP_FRAMES = 1  # Process every N frames (1 = every frame, 2 = every other frame)

print(f"\no Input video: {INPUT_VIDEO}")
print(f"o Output will be saved to: {OUTPUT_DIR}/{OUTPUT_VIDEO}")

# ============================================================================
# INITIALIZE SYSTEM
# ============================================================================

print("\n[1/4] Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL)
    print(f"- Model loaded: {YOLO_MODEL}")
    print(f"- Confidence Threshold = {CONFIDENCE_THRESHOLD}")
except Exception as e:
    print(f"- Error loading model: {e}")
    exit(1)

print("\n[2/4] Opening video...")
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print(f"Error: Could not open video file '{INPUT_VIDEO}'")
    print("- Make sure the video file is in the same folder as this script")
    print("- Or change the desire video file to 'traffic.mp4'")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"- Video opened: {frame_width}x{frame_height} @ {fps}fps")
print(f"- Total frames: {total_frames} (~{total_frames/fps:.1f} seconds)")

# Calculate actual line position
line_x1 = int(LINE_POSITION[0] * frame_width)
line_y1 = int(LINE_POSITION[1] * frame_height)
line_x2 = int(LINE_POSITION[2] * frame_width)
line_y2 = int(LINE_POSITION[3] * frame_height)

print(f"\n[3/4] Initiate LineCounter...")
counter = LineCounter(
    line_start=(line_x1, line_y1),
    line_end=(line_x2, line_y2),
    cooldown=1.0
)
print(f"- Counting line set from ({line_x1},{line_y1}) to ({line_x2},{line_y2})")

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Initialize video writer
output_path = Path(OUTPUT_DIR) / OUTPUT_VIDEO
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

print(f"\n[4/4] Starting video processing...")
print("="*70)

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

frame_count = 0
processed_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Skip frames if configured
    if frame_count % SKIP_FRAMES != 0:
        continue
    
    processed_count += 1
    current_time = time.time()
    
    # ========================================================================
    # YOLO DETECTION
    # ========================================================================
    
    # Run YOLO detection with tracking
    results = model.track(
        frame, 
        persist=True,  # Enable tracking
        conf=CONFIDENCE_THRESHOLD,
        classes=[2, 3, 5, 7],  # car=2, motorcycle=3, bus=5, truck=7
        verbose=False
    )
    
    # ========================================================================
    # PROCESS DETECTIONS
    # ========================================================================
    
    detections = results[0].boxes
    detected_count = 0
    
    if detections is not None and len(detections) > 0:
        for detection in detections:
            # Get bounding box
            bbox = detection.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get track ID (from ByteTrack built into YOLO)
            if detection.id is not None:
                track_id = int(detection.id.cpu().numpy()[0])
            else:
                continue  # Skip if no track ID
            
            # Calculate centroid
            centroid = Geometry.centroid_xyxy(bbox)
            
            # Update counter
            crossed = counter.update(track_id, centroid, current_time)
            
            # ================================================================
            # DRAW VISUALIZATION
            # ================================================================
            
            # Draw bounding box
            color = (0, 255, 0) if not crossed else (0, 255, 255)  # Green or yellow if just crossed
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID:{track_id}"
            cv2.putText(frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw centroid
            cx, cy = map(int, centroid)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            
            detected_count += 1
    
    # ========================================================================
    # DRAW UI OVERLAY
    # ========================================================================
    
    # Draw counting line
    cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), 
            (0, 0, 255), 3)  # Red line
    
    # Draw semi-transparent info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw statistics
    stats = counter.get_statistics()
    
    y_offset = 35
    cv2.putText(frame, f"Cars IN:  {stats['count_in']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Cars OUT: {stats['count_out']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"TOTAL:    {stats['total']}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"Detected: {detected_count}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # ========================================================================
    # WRITE OUTPUT & DISPLAY
    # ========================================================================
    
    # Write frame to output video
    out.write(frame)
    
    # Show live preview if enabled
    if SHOW_LIVE_PREVIEW:
        cv2.imshow('Car Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Stopped by user")
            break
    
    # Print progress
    if processed_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        elapsed = time.time() - start_time
        processing_fps = processed_count / elapsed
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
            f"Detected: {detected_count} | Total counted: {stats['total']} | "
            f"FPS: {processing_fps:.1f}")
    
    # Cleanup old tracks periodically
    if frame_count % 100 == 0:
        counter.cleanup_old_tracks(current_time)

# ============================================================================
# CLEANUP & FINAL STATISTICS
# ============================================================================

cap.release()
out.release()
cv2.destroyAllWindows()

processing_time = time.time() - start_time

print("\n" + "="*70)
print("✅ PROCESSING COMPLETE!")
print("="*70)

final_stats = counter.get_statistics()

print(f"\n📊 Final Statistics:")
print(f"   Cars IN:          {final_stats['count_in']}")
print(f"   Cars OUT:         {final_stats['count_out']}")
print(f"   TOTAL COUNTED:    {final_stats['total']}")
print(f"   Cars per minute:  {final_stats['cars_per_minute']:.1f}")

print(f"\n⏱️  Processing Info:")
print(f"   Total frames:     {total_frames}")
print(f"   Processed frames: {processed_count}")
print(f"   Processing time:  {processing_time:.1f} seconds")
print(f"   Processing FPS:   {processed_count/processing_time:.1f}")

print(f"\n📁 Output saved to:")
print(f"   {output_path.absolute()}")

print("\n" + "="*70)
print("🎉 Done! Open the output video to see the results!")
print("="*70)

# Optional: Determine traffic level (RED/YELLOW/GREEN)
cars_per_minute = final_stats['cars_per_minute']
if cars_per_minute < 20:
    traffic_level = "GREEN - Low Traffic"
elif cars_per_minute < 30:
    traffic_level = "YELLOW - Medium Traffic"
else:
    traffic_level = "RED - High Traffic"

print(f"\n🚦 Traffic Level: {traffic_level}")
print(f"   ({cars_per_minute:.1f} cars/minute)")