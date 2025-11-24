# Multi_Gesture_Control_Car_Racing_Game_Model
# 🏎️ F1 Gesture Racer Pro - Enhanced Edition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An immersive **3D gesture-controlled F1 racing game** that combines **real-time hand gesture recognition**, **facial emotion detection**, and **eye blink controls** for a revolutionary gaming experience!

## ✨ Features

### 🎮 Gesture Controls
- **🖐️ Palm Tilt** - Tilt your open palm left/right to change lanes
- **👍 Thumbs Up** - Activate turbo boost for maximum speed
- **✊ Fist** - Brake to slow down and avoid obstacles
- **👎 Thumbs Down** - Decline quit dialog and continue racing

### 👁️ Blink Controls
- **Single Blink** - Start race from menu or restart after game over
- **Double Blink** - Open quit menu during gameplay

### 🎭 Emotion Detection
The game features **real-time emotion analysis** that adapts commentary based on your facial expressions:
- **😊 Happy** - Encouraging and joyful commentary
- **😤 Angry** - Aggressive and intense feedback
- **😢 Sad** - Supportive and motivating messages
- **😲 Surprise** - Exciting reactions
- **🎯 Focused** - Professional racing commentary

### 🎨 Visual Effects
- **Neon-style UI** with modern design
- **Particle systems** for sparks, boosts, and explosions
- **Dynamic speed lines** and motion blur
- **Glowing effects** and smooth animations
- **Real-time camera preview** with gesture indicators

### 🎤 Voice Commentary
- **Text-to-speech** announces your score, milestones, and emotional state
- **Context-aware** feedback based on performance
- **Milestone announcements** (100, 200, 300, 500, 750, 1000 points)

## 📋 Requirements

### System Requirements
- **Python 3.8+**
- **Webcam** (for gesture and emotion detection)
- **Audio output** (for voice commentary)

### Python Dependencies
```bash
pip install opencv-python mediapipe pygame numpy pyttsx3 deepface
```

### Additional Requirements for Emotion Detection
DeepFace requires TensorFlow:
```bash
pip install tensorflow
```

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/f1-gesture-racer.git
cd f1-gesture-racer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add audio assets (optional):**
Create an `assets` folder and add the following sound files:
- `turbo-flutter-336362.mp3` - Boost sound
- `car-crash-sound-effect-376874.mp3` - Crash sound
- `background-music-224633.mp3` - Background music

4. **Run the game:**
```bash
python main_voice_commentary.py
```

## 🎮 How to Play

### Starting the Game
1. Launch the game and position yourself in front of the webcam
2. **Blink once** to start racing
3. Use hand gestures to control your F1 car

### Controls Summary
| Gesture | Action | Description |
|---------|--------|-------------|
| 🖐️ Tilt Palm Left/Right | Lane Change | Move between 3 lanes |
| 👍 Thumbs Up | Turbo Boost | Activate speed boost |
| ✊ Fist | Brake | Slow down |
| 👀 Single Blink | Start/Restart | Begin race or restart after crash |
| 👀👀 Double Blink | Quit Menu | Open quit dialog |
| 👎 Thumbs Down | Cancel Quit | Continue racing |

### Gameplay Tips
- **Lane management** - Stay in different lanes to avoid obstacles
- **Boost strategically** - Use thumbs up for speed bursts
- **Near misses** - Get close to obstacles without hitting for bonus points
- **Emotion matters** - Your facial expressions trigger unique commentary!
- **Milestone bonuses** - Reach score milestones for special announcements

### Obstacles
- **🚧 Traffic Cones** - Standard obstacles
- **🚧 Barriers** - Wider obstacles
- **🛢️ Oil Spills** - Slippery hazards

## 🏆 Scoring System

- **+10 points** per obstacle avoided
- **Near miss bonus** - Extra points for close calls
- **Perfect dodge streak** - Consecutive perfect dodges increase score multiplier
- **Emotion bonuses** - Special commentary based on your mood

### Rankings
- **0-200 points**: 🌟 Great Start!
- **200-500 points**: ⚡ Skilled Racer!
- **500-1000 points**: 🔥 Advanced Driver!
- **1000+ points**: 💎 Expert Racer!

## 🛠️ Technical Details

### Technologies Used
- **OpenCV** - Webcam capture and image processing
- **MediaPipe** - Hand gesture and face mesh detection
- **DeepFace** - Facial emotion recognition
- **Pygame** - Game engine and rendering
- **pyttsx3** - Text-to-speech voice commentary
- **NumPy** - Mathematical operations

### Key Features Implementation
- **Hand Tracking** - MediaPipe Hands with custom gesture recognition
- **Face Mesh** - 468 facial landmarks for blink detection
- **Emotion AI** - DeepFace with smoothing and confidence thresholding
- **Particle Systems** - Custom particle effects for visual feedback
- **Audio Manager** - Pygame mixer for sound effects and music

## 🎯 Performance Optimization

- **60 FPS target** with dynamic frame rate management
- **Gesture smoothing** with history-based detection
- **Efficient emotion processing** (1.2s intervals)
- **Optimized collision detection** with spatial partitioning
- **Adaptive difficulty** based on score progression

## 🐛 Troubleshooting

### Camera Issues
- Ensure webcam permissions are granted
- Check if camera is being used by another application
- Try changing camera index in code: `cv2.VideoCapture(0)` → `cv2.VideoCapture(1)`

### Emotion Detection Not Working
- Verify DeepFace installation: `pip install deepface tensorflow`
- Ensure good lighting conditions for face detection
- Position face clearly in camera view

### Audio Issues
- Check system audio settings and volume
- Verify pyttsx3 installation: `pip install pyttsx3`
- For Linux users: `sudo apt-get install espeak`

### Gesture Recognition Issues
- Ensure hand is well-lit and visible
- Keep hand within camera frame
- Make clear, distinct gestures
- Adjust detection confidence in code if needed

## 🔧 Configuration

You can customize various parameters in the code:

### Gesture Sensitivity
```python
# In GestureDetector.__init__()
min_detection_confidence=0.7  # Lower for easier detection (0.5-0.9)
min_tracking_confidence=0.7   # Lower for smoother tracking
```

### Blink Threshold
```python
# In GestureDetector.__init__()
self.blink_threshold = 0.20  # Lower for easier blink detection
```

### Game Difficulty
```python
# In RaceMode.__init__()
self.game_speed = 4  # Starting speed (3-6 recommended)
self.car.max_speed = 20  # Maximum speed (15-25 recommended)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 🙏 Acknowledgments

- **MediaPipe** team for excellent hand/face tracking
- **DeepFace** for emotion detection capabilities
- **Pygame** community for game development framework
- **OpenCV** for computer vision tools

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Email: guthikondapragna@gmail.com

## 🎮 Screenshots

*Add your game screenshots here*

---

**Made with ❤️ by Guthikonda Pragna**

⭐ Star this repo if you enjoyed the game!
