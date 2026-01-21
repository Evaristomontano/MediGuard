# ErgoGuard: Dynamic Posture Monitor

## Overview

ErgoGuard is a real-time posture monitoring tool built using computer vision (MediaPipe Pose). It detects and alerts you when you adopt poor working posture.

It analyzes two key metrics to prevent common issues like "Tech Neck":

1.  Neck Angle: Detects if your head is tilted forward or down.
2.  Protrusion (Z-Axis): Detects if your head is too close to the screen.

The system uses dynamic calibration to set a personalized threshold based on your perfect posture, ensuring high accuracy.

##  Setup and Installation

You need Python 3.8+ to run this project.

### Dependencies Installation

Install all required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt