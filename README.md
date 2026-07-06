# AlphaSwift — Autonomous Vision-Inertial Navigation in GNSS-Denied Environments

Final Year Project · Multimedia University · 2025/2026
Ryan Tung Tze Yang · Supervised by Dr Lo Yew Chiong · Moderated by Dr Cham Chin Leei · Industrial Moderator Prathiv Rao A/L Apparao

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
