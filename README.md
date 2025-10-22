# OBS People Detection System

A real-time people detection system that monitors OBS Virtual Camera or webcam feeds using YOLOv8 and automatically saves frames when people are detected.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- 🎥 **Real-time Detection**: Monitors video feed at 10 FPS
- 👤 **Person Detection**: Uses YOLOv8 nano model for accurate people detection
- 💾 **Auto-Save**: Automatically saves frames containing people
- 📊 **Live Preview**: Visual feedback with bounding boxes and confidence scores
- 🔧 **Smart Source Detection**: Automatically finds OBS Virtual Camera or webcam
- 📈 **Performance Stats**: Displays real-time FPS and detection statistics

## Demo

The system displays:
- Green bounding boxes around detected people
- Confidence scores for each detection
- Real-time status indicators
- Frame count and detection statistics

## Requirements

- Python 3.8 or higher
- Webcam or OBS Virtual Camera
- ~100MB disk space for model

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/obs-people-detector.git
cd obs-people-detector
```

2. **Install dependencies**
```bash
pip install opencv-python ultralytics numpy
```

3. **Run the script**
```bash
python detect_people.py
```

The YOLOv8 model will be downloaded automatically on the first run.

## Usage

### With OBS Studio

1. Open OBS Studio
2. Go to **Tools → Start Virtual Camera**
3. Run the script:
```bash
python detect_people.py
```

### With Webcam

Simply run the script - it will automatically detect your webcam:
```bash
python detect_people.py
```

### Controls

- **Press 'q'**: Quit the application
- **Ctrl+C**: Force quit

## Configuration

Edit these variables at the top of the script to customize behavior:

```python
OUTPUT_DIR = "detected_people"         # Output directory for saved frames
FPS = 10                               # Processing frame rate
CONFIDENCE_THRESHOLD = 0.5             # Detection confidence (0.0-1.0)
```

## Output

Detected frames are saved to the `detected_people/` directory with the following format:

```
person_20241020_153045_000123.jpg
       └─ date   └─ time  └─ frame number
```

## Troubleshooting

### No video source found
- **For OBS**: Make sure Virtual Camera is started (Tools → Start Virtual Camera)
- **For Webcam**: Check if another application is using the camera
- Try different camera indices by modifying the search range

### Low detection accuracy
- Increase `CONFIDENCE_THRESHOLD` to reduce false positives
- Decrease `CONFIDENCE_THRESHOLD` to detect more people (may include false positives)
- Ensure proper lighting in the video feed

### Performance issues
- Reduce `FPS` value for slower systems
- The YOLOv8 nano model is optimized for speed - consider upgrading to YOLOv8s for better accuracy if you have a GPU

## Dependencies

- **OpenCV**: Video capture and processing
- **Ultralytics**: YOLOv8 object detection
- **NumPy**: Numerical operations

## How It Works

1. **Video Capture**: Connects to OBS Virtual Camera or webcam
2. **Detection**: Each frame is analyzed using YOLOv8 for person detection
3. **Annotation**: Detected people are highlighted with bounding boxes
4. **Saving**: Frames containing people are automatically saved
5. **Display**: Live preview shows detection results in real-time

## Performance

- **Processing Speed**: ~10 FPS on modern CPUs
- **Model Size**: ~6MB (YOLOv8 nano)
- **Accuracy**: 50%+ confidence threshold by default
- **Memory Usage**: ~300-500MB

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Multiple person tracking
- [ ] Face blurring for privacy
- [ ] Video output instead of frame saves
- [ ] Web dashboard for monitoring
- [ ] Custom detection zones
- [ ] Email/SMS alerts

## License

MIT License - feel free to use and modify as needed.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [OpenCV](https://opencv.org/) - Computer vision library

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Note**: This system is designed for legitimate monitoring purposes only. Ensure you comply with local privacy laws when using video surveillance systems.
