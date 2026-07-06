#!/usr/bin/env python3
"""
aruco_pose_node.py
------------------
- Detects ArUco marker continuously in a camera thread
- Publishes pose at fixed 15Hz regardless of detection state
- When marker is detected:   publishes fresh pose, low covariance
- When marker is lost:       publishes last known pose, rising covariance
- When too stale (>2s):      clears pose — vision_sender will send max-covariance keepalive
- Fixes yaw wrap discontinuity with atan2(sin, cos) normalization
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool, Float32
import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation
import threading
import time
import math

CAMERA_MATRIX = np.array([
    [822.317,   0.0,     319.495],
    [  0.0,   822.317,   242.502],
    [  0.0,     0.0,       1.0  ]
])
DIST_COEFFS = np.array([[-0.0449369, 1.17277, 0.0, 0.0, -3.63244]])

MARKER_SIZE = 0.260          # UPDATE TO 0.300 when A3 marker printed

COV_FRESH   = 0.01
COV_STALE   = 0.50
STALE_LIMIT = 2.0

# Adaptive covariance parameters
COV_MIN       = 0.01    # best detection quality
COV_MAX_DET   = 0.30    # worst detection quality (still detected)
AREA_MIN      = 8000.0  # minimum area at max flying height (~1.2m)
AREA_MAX      = 25000.0 # maximum area at min flying height (~0.5m)


class ArucoPoseNode(Node):
    def __init__(self):
        super().__init__("aruco_pose_node")

        self.pose_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/aruco/pose_with_covariance", 10)
        self.pose_pub     = self.create_publisher(
            PoseStamped, "/mavros/vision_pose/pose", 10)
        self.mocap_pub    = self.create_publisher(
            PoseStamped, "/mavros/mocap/pose", 10)
        self.detected_pub = self.create_publisher(Bool,    "/aruco/marker_detected", 10)
        self.cov_pub      = self.create_publisher(Float32, "/aruco/covariance", 10)

        # Per-marker state dict — keys are marker IDs
        # Each value: {'latest_pose', 'last_detected_time', 'adaptive_cov',
        #              'pixel_area', 'detected_now'}
        self.markers = {}
        self.markers_lock = threading.Lock()
        # Backward compat fields (used by save snapshot logic)
        self.latest_pose        = None
        self.last_detected_time = None
        self.detected           = False
        self.latest_adaptive_cov = COV_MIN
        self.latest_pixel_area = 0.0
        self.filtered_yaw       = None
        self.yaw_init_count     = 0
        self.YAW_ALPHA          = 0.05
        self.lock               = threading.Lock()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera!")
            return

        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.adaptiveThreshWinSizeMin    = 3
        self.aruco_params.adaptiveThreshWinSizeMax    = 23
        self.aruco_params.adaptiveThreshWinSizeStep   = 10
        self.aruco_params.minMarkerPerimeterRate      = 0.03
        self.aruco_params.maxMarkerPerimeterRate      = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.05

        self.running    = True
        self.cam_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.cam_thread.start()

        self.timer = self.create_timer(1.0 / 15.0, self.publish_callback)

        self.get_logger().info("ArUco Pose Node started!")
        self.get_logger().info(f"Marker size: {MARKER_SIZE}m")
        self.get_logger().info("Publishing to /mavros/vision_pose/pose + /aruco/pose_with_covariance")

    def camera_loop(self):
        while self.running:
            self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS)

                # Track which markers were detected this frame
                detected_ids_this_frame = set()

                # Loop through ALL detected markers
                # Track which markers were detected this frame
                detected_ids_this_frame = set()

                # Loop through ALL detected markers
                for marker_idx in range(len(ids)):
                    marker_id = int(ids[marker_idx][0])
                    rvec = rvecs[marker_idx][0]
                    tvec = tvecs[marker_idx][0]
                    detected_ids_this_frame.add(marker_id)

                    ned_x = float(tvec[1])
                    ned_y = -float(tvec[0])
                    dist_to_marker = math.sqrt(ned_x**2 + ned_y**2)

                    # Save snapshot frames for FYP report (marker A / ID 0 only)
                    if marker_id == 0:
                        snap_frame = frame.copy()
                        aruco.drawDetectedMarkers(snap_frame, corners, ids)
                        label = f"x={ned_x:+.3f} y={ned_y:+.3f} dist={dist_to_marker:.3f}m"
                        cv2.putText(snap_frame, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        if not hasattr(self, '_centering_saved'):
                            self._centering_saved = False
                            self._locked_saved = False
                        if not self._centering_saved:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fn = f"/root/ros2_ws/logs/frame_centering_{ts}.jpg"
                            cv2.imwrite(fn, snap_frame)
                            self.get_logger().info(f"Saved CENTERING frame: {fn}")
                            self._centering_saved = True

                        if dist_to_marker < 0.08 and not self._locked_saved:
                            ts = time.strftime("%Y%m%d_%H%M%S")
                            fn = f"/root/ros2_ws/logs/frame_locked_{ts}.jpg"
                            cv2.putText(snap_frame, "LOCKED", (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            cv2.imwrite(fn, snap_frame)
                            self.get_logger().info(f"Saved LOCKED frame: {fn}")
                            self._locked_saved = True

                    # Adaptive covariance based on marker pixel area
                    marker_corners = corners[marker_idx][0]
                    pixel_area = float(cv2.contourArea(marker_corners))
                    quality = (pixel_area - AREA_MIN) / (AREA_MAX - AREA_MIN)
                    quality = max(0.0, min(1.0, quality))
                    adaptive_cov = COV_MAX_DET - quality * (COV_MAX_DET - COV_MIN)

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp    = self.get_clock().now().to_msg()
                    # Encode marker ID in frame_id
                    pose_msg.header.frame_id = f"aruco_id_{marker_id}"
                    pose_msg.pose.position.x = float(tvec[1])
                    pose_msg.pose.position.y = -float(tvec[0])
                    pose_msg.pose.position.z = -float(tvec[2])

                    R, _ = cv2.Rodrigues(rvec)
                    raw_yaw = np.arctan2(R[1, 0], R[0, 0]) + np.pi
                    raw_yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
                    if self.filtered_yaw is None:
                        self.filtered_yaw = raw_yaw
                        self.yaw_init_count = 0
                    if self.yaw_init_count < 30:
                        alpha = 0.5
                        self.yaw_init_count += 1
                    else:
                        alpha = self.YAW_ALPHA
                    diff = math.atan2(math.sin(raw_yaw - self.filtered_yaw),
                                      math.cos(raw_yaw - self.filtered_yaw))
                    self.filtered_yaw = self.filtered_yaw + alpha * diff
                    yaw = self.filtered_yaw

                    q = Rotation.from_euler("z", yaw).as_quat()
                    pose_msg.pose.orientation.x = float(q[0])
                    pose_msg.pose.orientation.y = float(q[1])
                    pose_msg.pose.orientation.z = float(q[2])
                    pose_msg.pose.orientation.w = float(q[3])

                    # Update per-marker state
                    with self.markers_lock:
                        self.markers[marker_id] = {
                            'latest_pose': pose_msg,
                            'last_detected_time': time.time(),
                            'adaptive_cov': adaptive_cov,
                            'pixel_area': pixel_area,
                            'detected_now': True,
                        }

                    # Backward compat for marker A (ID 0)
                    if marker_id == 0:
                        with self.lock:
                            self.latest_pose        = pose_msg
                            self.latest_adaptive_cov = adaptive_cov
                            self.latest_pixel_area = pixel_area
                            self.last_detected_time = time.time()
                            self.detected           = True

                # Mark non-detected known markers as stale this frame
                with self.markers_lock:
                    for known_id in self.markers:
                        if known_id not in detected_ids_this_frame:
                            self.markers[known_id]['detected_now'] = False
            else:
                # No markers in frame at all
                with self.markers_lock:
                    for known_id in self.markers:
                        self.markers[known_id]['detected_now'] = False
                with self.lock:
                    self.detected = False

    def publish_callback(self):
        now = time.time()

        # Snapshot all markers we know about
        with self.markers_lock:
            known_markers = dict(self.markers)

        any_detected = False
        for marker_id, state in known_markers.items():
            pose = state['latest_pose']
            detected_now = state['detected_now']
            last_time = state['last_detected_time']

            if pose is None:
                continue

            if detected_now:
                covariance = state['adaptive_cov']
                any_detected = True
            else:
                age = now - last_time if last_time else STALE_LIMIT
                if age >= STALE_LIMIT:
                    # Clear this marker — won't publish until re-detected
                    with self.markers_lock:
                        if marker_id in self.markers:
                            self.markers[marker_id]['latest_pose'] = None
                    self.get_logger().warn(
                        f"Marker ID {marker_id} lost >2s — clearing pose.",
                        throttle_duration_sec=2.0)
                    continue
                staleness  = age / STALE_LIMIT
                covariance = COV_FRESH + (COV_STALE - COV_FRESH) * staleness

            # Publish this marker's pose
            pose.header.stamp = self.get_clock().now().to_msg()
            # frame_id already set to "aruco_id_N" in camera_loop

            pcov_msg           = PoseWithCovarianceStamped()
            pcov_msg.header    = pose.header
            pcov_msg.pose.pose = pose.pose
            cov_diag = [0.0] * 36
            for i in range(3):
                cov_diag[i*6+i] = float(covariance)
            for i in range(3, 6):
                cov_diag[i*6+i] = 0.01
            pcov_msg.pose.covariance = cov_diag
            self.pose_cov_pub.publish(pcov_msg)

            # Marker A (ID 0) also publishes to MAVROS topics for EKF
            if marker_id == 0:
                self.pose_pub.publish(pose)
                self.mocap_pub.publish(pose)

            cov_msg      = Float32()
            cov_msg.data = float(covariance)
            self.cov_pub.publish(cov_msg)

            self.get_logger().info(
                f"ID:{marker_id} X:{pose.pose.position.x:.3f} Y:{pose.pose.position.y:.3f} "
                f"Z:{pose.pose.position.z:.3f}m | cov:{covariance:.3f} | "
                f"area:{state['pixel_area']:.0f}px | det:{'Y' if detected_now else 'N(stale)'}",
                throttle_duration_sec=0.5)

        # Publish overall "any marker detected" flag
        detected_msg      = Bool()
        detected_msg.data = any_detected
        self.detected_pub.publish(detected_msg)

    def destroy_node(self):
        self.running = False
        self.cam_thread.join(timeout=2.0)
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
