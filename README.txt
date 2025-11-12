"""
Car Counter System
Integrates YOLO detection + ByteTrack + LineCounter
Processes video and creates output with car counting visualization
"""

### Core Functionality
- Real-time Vehicle Detection - Uses YOLOv8 nano model for fast, accurate car detection
- Bidirectional Counting - Tracks vehicles entering and exiting with separate counts
- Live Preview Mode - Optional real-time visualization
- Video Output - Generates output video with bounding boxes and statistics

### Methodology

============================================================================
 Install the Dependencies
============================================================================

Within the same folder open terminal and cd to correct folder and type command:

$ pip install ultralytics
$ pip install opencv-python
$ pip install numpy

Or use this command:

$ pip install ultralytics opencv-python numpy



============================================================================
 Prepare traffic video
============================================================================

Download from source like website https://www.pexels.com/search/traffic

Place your traffic video in the project folder and name it `traffic.mp4`
NOTE: can put as many video in folder BUT the desire video have to name `traffic.mp4`

**Recommended video specs:**
- Format: MP4
- Duration: 10-60 seconds for testing (long video = long processing)
- Content: Clear view of traffic (highway, not too small)

(Update configuration in `car_counter_main.py`)
# Input/Output
INPUT_VIDEO = "traffic.mp4"           # Your video name file
OUTPUT_DIR = "output"                 # Output folder
OUTPUT_VIDEO = "counted_video.mp4"    # Output filename



============================================================================
 Change the setting accordingly in `car_counter_main.py`
============================================================================

In file `car_counter_main.py`

- LINE_POSITION | for line position, can uncomment the provided line in the
- YOLO_MODEL = "yolov8n.pt" | can be using other model by changing the string
- CONFIDENCE_THRESHOLD | range (0.0 to 1.0) default set to 0.3

** CONFIDENCE_THRESHOLD selection range **
Set it Too LOW (0.0 or 0.1)
Pros: won't miss real detection, very sensitive
Cons: tons of false positive (detect things that aren't there)
Example: label Shadow as a Car, group 2 car as 1 box

Set it Too HIGH (0.9 or 1.0)
Pros: few false positive
Cons: will miss many real detection (become overly strict)
Example: ignore small or blurry cars bc it's not 95% sure



============================================================================
 Live video setting in `car_counter_main.py`
============================================================================

Display the real-time counting process with separate window(slower)
Controls:
- Watch real-time counting in window
- Press 'Q' to stop processing early

SHOW_LIVE_PREVIEW = False  # Set True to see processing in real-time
SKIP_FRAMES = 1  # Process every N frames



============================================================================
 Run the script
============================================================================

Run with default settings:

$ python car_counter_main.py

running the scripte for the first time generate
- YOLO file model eg. `yolov8n.pt`
- Cache folder `_pycache_`
- Created `output` folder with `counted_video.mp4` output

Output:
- `counted_video.mp4` in `output` folder
- Manual open video with  File Explorer



============================================================================
 How it work
============================================================================

1. Input video
2. YOLO model detection - detect vehicle, give box
3. ByteTrack - get unique tracking ID
4. Geometry module - calculate centroid
5. Line Counter - check line crossing
6. Visualization - draw box and status
7. Output Video