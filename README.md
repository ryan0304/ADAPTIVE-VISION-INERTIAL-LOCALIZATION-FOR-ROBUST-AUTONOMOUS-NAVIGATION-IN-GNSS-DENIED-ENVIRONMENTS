# AlphaSwift — Autonomous Vision-Inertial Navigation in GNSS-Denied Environments

Final Year Project · Multimedia University · 2025/2026
Tung Tze Yang · Supervised by Dr Lo Yew Chiong · Moderated by Dr Cham Chin Leei · Industrial Moderator Prathiv Rao A/L Apparao

Autonomous hexacopter navigation using ArUco fiducial markers, optical flow,
and a rangefinder for GPS-denied indoor flight. Achieves 0.04–0.06 m RMSE
per centring cycle on a full A→B→C→land mission.

## Repository layout

- `src/aruco_pose/` — ROS2 package: ArUco detection, adaptive covariance, pose publishing
- `vision_sender/` — Mission controller: state machine, MAVLink IO, visual servoing
- `docs/` — Technical documentation (to be added)
- `tools/` — Analysis tools (to be added)
- `presentation/` — FYP2 defence deck (to be added)

## Stack

- Hardware: F550 hex + AlphaSwift FC Mini + Jetson Nano + Logitech C270 + MTF-01
- Software: ArduCopter 4.6.3 · ROS2 Humble · Ubuntu 22.04 · Docker
- Comms: pymavlink direct-serial to FC, no MAVROS in the data path

## Running the System

The system runs as two ROS2 nodes inside a Docker container on the Jetson Nano. Each node runs in its own terminal — the perception node in Terminal 1, the mission controller in Terminal 2.

### Prerequisites

- Jetson Nano flashed with Ubuntu 22.04
- Docker installed and the `my_drone_run:latest` image built
- USB camera exposed at `/dev/video0`
- Flight controller connected via FTDI, pinned to `/dev/ttyFCU` by udev rule
- ArUco markers printed at 0.260 m physical size (matching `MARKER_SIZE` in `aruco_pose_node.py`)
- ArduCopter 4.6.3 flashed to the AlphaSwift FC Mini with EK3_SRC3 configured (see docs)

### Terminal 1 — Start the container and run the perception node

```bash
docker run -it --rm --name my_drone_run \
    --net=host --ipc=host \
    --device /dev/video0:/dev/video0 \
    --device /dev/ttyFCU:/dev/ttyFCU \
    -e DISPLAY=:1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/ros2_ws:/root/ros2_ws \
    --entrypoint bash \
    my_drone_run:latest

# Inside the container:
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
mkdir -p /root/ros2_ws/logs
ros2 run aruco_pose aruco_pose_node 2>&1 | tee /root/ros2_ws/logs/aruco_$(date +%Y%m%d_%H%M%S).log
```

The perception node opens the camera, starts ArUco detection at ~15 Hz, and publishes `PoseWithCovarianceStamped` messages on `/aruco/pose_with_covariance`.

### Terminal 2 — Attach to the same container and run the mission controller

```bash
docker exec -it my_drone_run bash

# Inside the container:
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
mkdir -p /root/ros2_ws/logs
python3 /root/ros2_ws/vision_sender.py 2>&1 | tee /root/ros2_ws/logs/vision_sender_$(date +%Y%m%d_%H%M%S).log
```

The mission controller connects to the FC over `/dev/ttyFCU`, subscribes to the perception topic, and waits for a command on stdin.

### Operator commands (Terminal 2 stdin)

Once `vision_sender.py` is running, type one of these:

- `arm` — standard ArduCopter arm sequence, then execute the mission
- `forcearm` — force arm (bypass prearm checks), then execute the mission
- `land` — issue landing command immediately
- `abort` — disarm immediately
- `status` — print current state, position, marker detection status

### Log outputs

All logs are written under `/root/ros2_ws/logs/`:

- `aruco_YYYYMMDD_HHMMSS.log` — perception node stdout
- `vision_sender_YYYYMMDD_HHMMSS.log` — mission controller stdout
- `rmse_YYYYMMDD_HHMMSS.txt` — per-centring-cycle RMSE files (one per marker hold)
- `frame_centering_*.jpg`, `frame_locked_*.jpg` — camera snapshots (Marker A only, once per session)

### Recovery from mid-mission failure

If either node crashes or gets stuck, the safest path is:

1. Terminal 2: `Ctrl+C` to kill `vision_sender.py`
2. Terminal 1: `Ctrl+C` to kill `aruco_pose_node`
3. Power-cycle the FCU (mandatory — the EKF's Z state can carry over otherwise, see docs)
4. Restart both nodes from the same commands above
