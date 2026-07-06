#!/usr/bin/env python3
"""
vision_sender.py — Guided Mode Visual Servoing
AlphaSwift FYP

Architecture (from flight log analysis):
- Stay in Guided mode 4 entire flight (optical flow holds XY)
- Use LOCAL_NED absolute position target (not body offset)
  Body offset accumulates — absolute position does not
- Read LOCAL_POSITION_NED for current drone position
- Compute target = current_pos + marker_offset
- Send absolute target position to FCU

Phase B addition: anchor hold mode
- Once locked for 2s within deadband, capture position as anchor
- Hold at fixed anchor until marker drifts >12cm OR 10s elapsed
- Provides rock-solid hover for FYP mission Phase B

Arming:
- "arm"      -> normal arm, respects FCU prearm checks (default, safe)
- "forcearm" -> force arm, bypasses prearm checks (use deliberately only)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from pymavlink import mavutil
import threading
import time
import math
import os
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

SERIAL_PORT      = '/dev/ttyFCU'
BAUD_RATE        = 921600
TAKEOFF_ALT_M    = 1.2
HOLD_DURATION_S  = 10.0
DEADBAND_M       = 0.08
ANCHOR_RELEASE_M = 0.20
ANCHOR_DELAY_S   = 2.0
CONFIRM_FRAMES   = 3
COV_MAX_ACCEPT   = 0.25
K_P              = 0.15
HOVER_STABLE_S   = 5.5
# Centering active-approach: velocity control instead of absolute position
# target (avoids accumulating EKF pos_x/pos_y drift into the commanded
# target over many cycles — locked/anchor-hold phase below DEADBAND_M
# still uses position control since that's a one-time capture, not
# continuous accumulation).
CENTER_VEL_K_FAR  = 0.8    # m/s per meter of marker offset, dist > 0.15m
CENTER_VEL_K_NEAR = 0.55   # m/s per meter of marker offset, 0.10-0.15m
CENTER_VEL_MAX    = 0.35   # m/s cap on commanded centering velocity
CENTER_VEL_STALE_S = 0.25  # max age of x/y reading before holding instead
                            # of continuing to execute a velocity derived
                            # from outdated marker data (much tighter than
                            # the general 'detected' staleness window,
                            # since a continuously-executing velocity
                            # command on stale data keeps moving the
                            # drone long after the reading stopped
                            # being trustworthy — confirmed via flight log
                            # showing x/y frozen for ~0.9s while the drone
                            # kept translating on the last valid command)
CENTER_VEL_COV_MAX = 0.05  # much tighter than COV_MAX_ACCEPT (0.25) —
                            # for active velocity control we only trust
                            # genuinely fresh, low-covariance detections,
                            # not "good enough to count as detected" ones
CENTER_VEL_MAX_REPEATS = 2 # if x/y is bit-identical to the previous
                            # reading for more than this many consecutive
                            # cycles, treat it as a stale republish
                            # (aruco_pose_node.py's keepalive behavior)
                            # rather than a genuinely static marker, and
                            # hold instead of continuing to servo on it
NEAR_ZONE_ENTER_M  = 0.10  # enter near-zone hold below this
NEAR_ZONE_EXIT_M   = 0.12  # only resume active servoing above this —
                            # gap between enter/exit prevents a single
                            # noisy sample at the boundary from kicking
                            # the drone into active translation
# Phase C: multi-marker mission
NAV_DISTANCE_M   = 1.3   # how far to fly north for marker B (reference distance)
NAV_ARRIVAL_M    = 0.15  # used only as an early-exit if position data looks
                          # sane; not required for open-loop velocity nav
SEARCH_TIMEOUT_S = 15.0  # how long to look for B before giving up
# Velocity-based open-loop navigation (bypasses LOCAL_POSITION_NED dependency)
NAV_VEL_MPS      = 0.30  # forward (north) velocity command, m/s
NAV_NORTH_S      = NAV_DISTANCE_M / NAV_VEL_MPS  # time to cover NAV_DISTANCE_M
                          # at NAV_VEL_MPS (e.g. 4.5s for 1.35m @ 0.3m/s)
NAV_RAMP_DOWN_S  = 0.4   # final braking window: send zero-velocity hold
                          # before declaring arrival, to kill residual drift
NAV_TIMEOUT_MARGIN_S = 2.0  # safety margin ABOVE normal drive+brake time
NAV_TIMEOUT_S    = NAV_NORTH_S + NAV_RAMP_DOWN_S + NAV_TIMEOUT_MARGIN_S
                          # auto-scales with NAV_DISTANCE_M/NAV_VEL_MPS so
                          # changing distance can never make the timeout
                          # collide with or precede normal completion
# Heading-fix: one-time yaw alignment to marker A right before Phase C drive
HEADING_FIX_TOLERANCE_DEG = 1.0   # declare aligned within this many degrees
HEADING_FIX_K             = 0.8   # proportional gain, yaw_rate(rad/s) per rad error
HEADING_FIX_MAX_RATE      = 0.3   # rad/s cap, ~17 deg/s, keep correction gentle
HEADING_FIX_TIMEOUT_S     = 5.0   # give up and proceed to Phase C regardless
HEADING_FIX_STALE_S       = 1.0   # marker_yaw older than this is not trusted
LOG_DIR          = os.path.expanduser('~/ros2_ws/logs')

# Force-arm magic number (ArduPilot convention — bypasses prearm checks)
FORCE_ARM_MAGIC  = 21196

# ═══════════════════════════════════════════════════════
# STATES
# ═══════════════════════════════════════════════════════

STATE_IDLE=0; STATE_ARMING=1; STATE_ARMED_WAIT=2; STATE_TAKEOFF=3
STATE_HOVER=4; STATE_CENTERING=5; STATE_LOCKED=6; STATE_LANDING=7; STATE_DONE=8
# Phase C states
STATE_NAVIGATE_TO_B=9; STATE_SEARCHING_B=10
# Phase B->C handoff: one-time yaw alignment to marker A before driving forward
STATE_HEADING_FIX=11
# Phase D states: same pattern as Phase C, marker B -> marker C (ID 2)
STATE_NAVIGATE_TO_C=12; STATE_SEARCHING_C=13

STATE_NAMES={0:'IDLE',1:'ARMING',2:'ARMED_WAIT',3:'TAKEOFF',
             4:'HOVER',5:'CENTERING',6:'LOCKED',7:'LANDING',8:'DONE',
             9:'NAVIGATE_TO_B',10:'SEARCHING_B',11:'HEADING_FIX',
             12:'NAVIGATE_TO_C',13:'SEARCHING_C'}

# ═══════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════

class SS:
    _lock           = threading.Lock()
    state           = STATE_IDLE
    state_t         = time.time()
    armed           = False
    alt_m           = 0.0
    alt_last_t      = 0.0
    # EKF position (LOCAL_POSITION_NED)
    pos_x           = 0.0   # NED north
    pos_y           = 0.0   # NED east
    pos_z           = 0.0   # NED down
    ground_alt      = 0.0
    takeoff_sent    = False
    reacquire       = False
    pose_x          = 0.0
    pose_y          = 0.0
    pose_cov        = 9999.0
    pose_t          = None
    marker_yaw      = None   # radians, from ArUco rvec via quaternion
    marker_yaw_t    = None
    heading_fix_start_t = None
    # After heading-fix completes, a second full STATE_CENTERING
    # lock+hold+RMSE pass runs before driving to the next marker.
    # next_phase_c remembers which NAV state to enter once that
    # second pass completes, since current_target_id must NOT switch
    # until the drive actually starts (the second centering pass
    # still needs to track the marker we just aligned to, not the
    # next one).
    next_phase_c = None  # True -> NAVIGATE_TO_B, False -> NAVIGATE_TO_C
    confirmed       = 0
    rmse_samples    = []
    lock_start_t    = None
    last_tgt_n      = None
    last_tgt_e      = None
    lock_anchor_n   = None
    lock_anchor_e   = None
    # Recovery anchor: last good drone NED position when marker was fresh
    recovery_anchor_n = None
    recovery_anchor_e = None
    # Set true on marker re-acquisition to trigger fly-to-recovery
    recovery_pending = False
    recovery_start_t = None
    # Phase C: which ArUco ID we're currently targeting (0=A, 1=B)
    current_target_id = 0
    # Phase C: open-loop velocity navigation timing
    nav_start_t = None
    nav_start_pos_n = None   # pos_x at nav start, for logging/sanity check only
    nav_start_pos_e = None
    # Velocity estimate from consecutive pos_x/pos_y samples, for braking
    prev_pos_n      = None
    prev_pos_e      = None
    # Repeated-marker-pose detection: aruco_pose_node.py republishes the
    # LAST KNOWN pose with rising covariance when detection goes stale,
    # on the same publish schedule — so pose_t alone looks "fresh" even
    # when x/y haven't actually changed. Track consecutive identical
    # readings directly to catch this (confirmed via flight log showing
    # x/y frozen for ~0.9s while pose_t-based staleness check still
    # judged it fresh).
    prev_marker_x   = None
    prev_marker_y   = None
    marker_repeat_count = 0
    # Same repeat-detection pattern, applied to marker_yaw for
    # STATE_HEADING_FIX — same stale-republish issue can affect yaw
    # exactly like it did x/y (confirmed via flight log showing
    # heading-fix error frozen at -18.36deg for ~2.3s while myaw_t
    # (timestamp-only check) still judged it fresh).
    prev_marker_yaw = None
    marker_yaw_repeat_count = 0
    # Hysteresis for near-zone <-> active-servo boundary: without this,
    # a single noisy dist reading crossing 0.10m can kick the drone into
    # active velocity servoing for one cycle, imparting real momentum
    # that a subsequent dip back under 0.10m doesn't undo (velocity
    # commands have no automatic restoring force the way position
    # targets do) — confirmed via flight log showing dist oscillate
    # 0.095/0.107m near the boundary while never re-entering NEAR-CENTER
    # hold after the first active-servo cycle.
    in_active_servo = False
    prev_pos_t      = None
    est_vel_n       = 0.0
    est_vel_e       = 0.0
    # Near-zone active braking: True while we're in the one-shot braking
    # pulse triggered on first entry to the near zone with nonzero velocity
    braking_active  = False
    braking_start_t = None
    was_in_near_zone = False
    # Phase B done flag (don't redo Phase B if we re-enter CENTERING)
    phase_b_complete = False
    phase_c_complete = False
    # Arming: which mode was requested ('normal' or 'force')
    arm_mode = 'normal'
    # Last PreArm/STATUSTEXT failure reason seen from FCU (for visibility)
    last_prearm_msg = None

    @classmethod
    def set_state(cls, s, logger=None):
        with cls._lock:
            old = cls.state
            cls.state = s
            cls.state_t = time.time()
        msg = f'STATE: {STATE_NAMES[old]} → {STATE_NAMES[s]}'
        if logger: logger.info(msg)
        else: print(msg)

    @classmethod
    def time_in_state(cls):
        with cls._lock:
            return time.time() - cls.state_t

# ═══════════════════════════════════════════════════════
# MAVLINK HELPERS
# ═══════════════════════════════════════════════════════

def mav_arm(mav, force=False):
    """
    Arm the FCU.
    force=False (default): normal arm, FCU prearm checks apply (safe).
    force=True: bypasses prearm checks via magic param2 value.
                Only use when you've deliberately decided to override
                a known/understood check — not as the default path.
    """
    param2 = FORCE_ARM_MAGIC if force else 0
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, param2, 0, 0, 0, 0, 0)

def mav_disarm(mav, force=False):
    param2 = FORCE_ARM_MAGIC if force else 0
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, param2, 0, 0, 0, 0, 0)

def mav_guided(mav):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0)

def mav_takeoff(mav, alt):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, alt)

def mav_hold(mav):
    """Zero velocity hold — optical flow maintains XY position."""
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111000111,  # velocity only, all zero
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

def mav_goto_ned(mav, north, east, down):
    """
    Go to absolute NED position with YAW LOCKED (yaw_rate=0).
    Position only — velocity, accel, yaw ignored.
    yaw_rate=0 prevents FCU from auto-yawing toward target.
    """
    # Bitmask bits to IGNORE (1=ignore):
    # bit 0-2: position (use - all zero)
    # bit 3-5: velocity (ignore)
    # bit 6-8: accel (ignore)
    # bit 9: force flag (ignore)
    # bit 10: yaw (ignore - we don't want absolute yaw control)
    # bit 11: yaw_rate (USE - set yaw_rate=0, locks heading)
    # NOTE: bits 10/11 were previously swapped (0b0000101111111000),
    # which told the FCU to use an ABSOLUTE yaw target of 0 radians
    # instead of locking current heading via yaw_rate=0 — every
    # position command was silently also commanding a yaw snap to
    # whatever the EKF considers zero heading, fighting translation
    # with an unwanted rotational pull whenever actual heading
    # differed from that reference (confirmed via gyro drift
    # measurements earlier this session). This explains a
    # consistent directional drift bias independent of marker
    # position or gain tuning.
    type_mask = 0b0000011111111000
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        north, east, down,
        0, 0, 0,    # vx, vy, vz (ignored)
        0, 0, 0,    # ax, ay, az (ignored)
        0, 0)       # yaw (ignored), yaw_rate=0 (USED, locks heading)

def mav_velocity_ned(mav, vn, ve, vd, yaw_rate=0.0):
    """
    Command NED-frame velocity (open-loop), with optional yaw_rate.
    Used for Phase C open-loop navigation — bypasses any dependency
    on LOCAL_POSITION_NED being healthy/accurate. Also used by
    STATE_CENTERING for active-approach and anchor-hold, optionally
    with a live yaw correction (see STATE_CENTERING heading-fix
    integration) rather than always locking yaw_rate=0.
    vn/ve/vd in m/s, NED frame (vd negative = climbing).
    yaw_rate in rad/s, defaults to 0.0 (locks current heading) for
    backward compatibility with existing callers that don't pass it.
    """
    # Bitmask bits to IGNORE (1=ignore):
    # bit 0-2: position (ignore)
    # bit 3-5: velocity (USE)
    # bit 6-8: accel (ignore)
    # bit 9: force flag (ignore)
    # bit 10: yaw (ignore - we don't want absolute yaw control)
    # bit 11: yaw_rate (USE)
    # NOTE: same bit 10/11 swap bug as mav_goto_ned — see that
    # function's comment for full explanation. Fixed identically here.
    type_mask = 0b0000011111000111
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,    # x, y, z (ignored)
        vn, ve, vd, # vx, vy, vz (USED)
        0, 0, 0,    # ax, ay, az (ignored)
        0, yaw_rate) # yaw (ignored), yaw_rate (USED)

def mav_yaw_rate(mav, yaw_rate):
    """
    Command pure yaw rotation rate while holding zero XYZ velocity —
    used for one-time heading alignment to a marker (STATE_HEADING_FIX).
    yaw_rate in rad/s, positive = clockwise (matches MAVLink convention).
    """
    # Bitmask bits to IGNORE (1=ignore):
    # bit 0-2: position (ignore)
    # bit 3-5: velocity (USE — held at zero so the drone doesn't translate
    #          while rotating)
    # bit 6-8: accel (ignore)
    # bit 9: force flag (ignore)
    # bit 10: yaw (ignore — we command yaw_rate instead, not absolute yaw)
    # bit 11: yaw_rate (USE)
    # NOTE: bits 10/11 were previously swapped (0b0000100111000111),
    # which told the FCU to use absolute yaw=0 and IGNORE yaw_rate —
    # explaining why commanded cmd_rate had little/no visible effect.
    type_mask = 0b0000011111000111
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        0, 0, 0,        # x, y, z (ignored)
        0.0, 0.0, 0.0,  # vx, vy, vz (USED — zero, hold position while turning)
        0, 0, 0,        # ax, ay, az (ignored)
        0, yaw_rate)    # yaw (ignored), yaw_rate (USED)

def compute_yaw_correction(marker_yaw, marker_yaw_t, cov,
                            prev_yaw, repeat_count):
    """
    Shared proportional yaw-rate correction toward marker_yaw=0,
    used by both STATE_HEADING_FIX (one-time pre-Phase-C alignment)
    and STATE_CENTERING (continuous correction while approaching).

    Returns (yaw_rate, is_trustworthy, new_repeat_count). If the
    marker_yaw reading is stale (by timestamp, by covariance, or by
    being bit-identical to the previous reading for too many
    consecutive cycles — same repeat-detection pattern used for the
    centering x/y staleness guard, since aruco_pose_node.py
    republishes the last known orientation on the same schedule as
    position when detection goes stale) then is_trustworthy=False
    and yaw_rate=0.0, signaling the caller to hold rather than
    command a correction based on outdated data.
    """
    if marker_yaw is None or marker_yaw_t is None:
        return 0.0, False, 0
    age = time.time() - marker_yaw_t
    if age > HEADING_FIX_STALE_S:
        return 0.0, False, 0
    if cov > CENTER_VEL_COV_MAX:
        return 0.0, False, repeat_count
    if prev_yaw is not None and marker_yaw == prev_yaw:
        new_repeat_count = repeat_count + 1
    else:
        new_repeat_count = 0
    if new_repeat_count > CENTER_VEL_MAX_REPEATS:
        return 0.0, False, new_repeat_count
    cmd_rate = HEADING_FIX_K * marker_yaw
    cmd_rate = max(-HEADING_FIX_MAX_RATE, min(HEADING_FIX_MAX_RATE, cmd_rate))
    return cmd_rate, True, new_repeat_count

def mav_land(mav):
    """Send NAV_LAND command — direct landing action, more reliable than mode change."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,           # confirmation
        0,           # abort altitude (0 = no abort)
        0,           # land mode (0 = normal, opportunistic precision land)
        0, 0,        # empty / yaw angle
        0, 0,        # target lat, lon (0 = current position)
        0)           # target altitude (0 = land at ground)

# ═══════════════════════════════════════════════════════
# RMSE
# ═══════════════════════════════════════════════════════

def save_rmse(samples, logger):
    if not samples: return
    os.makedirs(LOG_DIR, exist_ok=True)
    fn = os.path.join(LOG_DIR,
         f'rmse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    arr = np.array(samples)
    rx = float(np.sqrt(np.mean(arr[:,0]**2)))
    ry = float(np.sqrt(np.mean(arr[:,1]**2)))
    rt = float(np.sqrt(np.mean(arr[:,0]**2 + arr[:,1]**2)))
    with open(fn,'w') as f:
        f.write(f'AlphaSwift FYP Phase 4 RMSE\n')
        f.write(f'Samples: {len(samples)}\n')
        f.write(f'RMSE X:     {rx:.4f}m\n')
        f.write(f'RMSE Y:     {ry:.4f}m\n')
        f.write(f'RMSE Total: {rt:.4f}m\n')
        for x,y in samples: f.write(f'{x:.4f},{y:.4f}\n')
    logger.info(f'RMSE saved → {fn}')
    logger.info(f'RMSE X={rx:.4f}m Y={ry:.4f}m Total={rt:.4f}m')

# ═══════════════════════════════════════════════════════
# ROS2 NODE
# ═══════════════════════════════════════════════════════

class VisionSender(Node):
    def __init__(self):
        super().__init__('vision_sender')
        os.makedirs(LOG_DIR, exist_ok=True)
        self.mav = None
        self.connected = False
        self._connect()
        self.sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/aruco/pose_with_covariance',
            self._pose_cb, 10)
        for t,n in [(self._hb_loop,'hb'),(self._recv_loop,'recv'),
                    (self._send_loop,'send'),(self._input_loop,'input'),
                    (self._stream_keepalive,'stream_keepalive')]:
            threading.Thread(target=t, daemon=True, name=n).start()
        self.get_logger().info('Vision sender ready — Guided mode | 15Hz')
        self.get_logger().info(
            f'Takeoff:{TAKEOFF_ALT_M}m K_P:{K_P} '
            f'Deadband:{DEADBAND_M}m AnchorRelease:{ANCHOR_RELEASE_M}m')
        self.get_logger().info(
            f'Phase C: NavDist:{NAV_DISTANCE_M}m '
            f'NavTimeout:{NAV_TIMEOUT_S}s SearchTimeout:{SEARCH_TIMEOUT_S}s')
        self.get_logger().info(
            'Commands: arm | forcearm | land | abort | status')
        self.get_logger().warn(
            'forcearm bypasses FCU prearm checks (EKF/IMU/GPS faults '
            'will NOT block it) — only use deliberately, never as default')

    def _connect(self):
        while not self.connected:
            try:
                self.get_logger().info('Connecting to FCU...')
                self.mav = mavutil.mavlink_connection(
                    SERIAL_PORT, baud=BAUD_RATE)
                msg = self.mav.recv_match(
                    type='HEARTBEAT', blocking=True, timeout=10)
                if msg and msg.get_srcSystem()==1:
                    self.mav.target_system    = 1
                    self.mav.target_component = 1
                    self.connected = True
                    self.get_logger().info('FCU connected!')
                    time.sleep(0.5)
                    mav_guided(self.mav)
                    self.get_logger().info('Mode: GUIDED')
                    self.mav.mav.request_data_stream_send(
                        self.mav.target_system,
                        self.mav.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1)
                else:
                    time.sleep(2)
            except Exception as e:
                self.get_logger().warn(f'Connect error: {e}')
                time.sleep(2)

    def _stream_keepalive(self):
        '''Aggressively request LOCAL_POSITION_NED at 10Hz to keep stream alive'''
        while True:
            try:
                if self.mav and self.connected:
                    # SET_MESSAGE_INTERVAL for LOCAL_POSITION_NED (msg id 32) at 100ms = 10Hz
                    self.mav.mav.command_long_send(
                        self.mav.target_system, self.mav.target_component,
                        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                        0, 32, 100000, 0, 0, 0, 0, 0)
                    # Also legacy request for redundancy
                    self.mav.mav.request_data_stream_send(
                        self.mav.target_system,
                        self.mav.target_component,
                        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                        10, 1)
            except Exception as e:
                pass
            time.sleep(0.5)  # Re-request every 500ms

    def _hb_loop(self):
        ctr = 0
        while True:
            try:
                if self.mav and self.connected:
                    self.mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    ctr += 1
                    if ctr >= 1:
                        ctr = 0
                        self.mav.mav.request_data_stream_send(
                            self.mav.target_system,
                            self.mav.target_component,
                            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                            10, 1)
                        self.mav.mav.request_data_stream_send(
                            self.mav.target_system,
                            self.mav.target_component,
                            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)
            except Exception as e:
                self.get_logger().warn(f'HB error: {e}')
                self.connected = False
                self._connect()
            time.sleep(0.5)  # 2Hz heartbeat (was 1Hz)

    def _recv_loop(self):
        while True:
            try:
                if not self.connected:
                    time.sleep(0.1); continue
                msg = self.mav.recv_match(blocking=True, timeout=0.1)
                if not msg: continue
                t = msg.get_type()
                if t == 'HEARTBEAT':
                    armed = bool(msg.base_mode &
                        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    with SS._lock: SS.armed = armed
                elif t == 'LOCAL_POSITION_NED':
                    with SS._lock:
                        SS.pos_x    = float(msg.x)   # NED north
                        SS.pos_y    = float(msg.y)   # NED east
                        SS.pos_z    = float(msg.z)   # NED down
                        SS.alt_m    = -float(msg.z)  # altitude
                        SS.alt_last_t = time.time()
                        if not hasattr(SS, '_pos_msg_count'):
                            SS._pos_msg_count = 0
                            SS._pos_msg_start = time.time()
                        SS._pos_msg_count += 1
                        # Print rate every 5 seconds
                        elapsed = time.time() - SS._pos_msg_start
                        if elapsed > 5.0:
                            rate = SS._pos_msg_count / elapsed
                            print(f'>>> LOCAL_POSITION_NED rate: {rate:.1f} Hz '
                                  f'({SS._pos_msg_count} in {elapsed:.1f}s) '
                                  f'last x={msg.x:+.3f} y={msg.y:+.3f}')
                            SS._pos_msg_count = 0
                            SS._pos_msg_start = time.time()
                elif t == 'STATUSTEXT':
                    text = msg.text if hasattr(msg, 'text') else str(msg)
                    low = text.lower()
                    # All FCU STATUSTEXT (PreArm, EKF, etc.) is captured
                    # silently for the 'status' command and arm-timeout
                    # error, but NOT printed here — view live FCU messages
                    # in Mission Planner's Messages tab instead, to avoid
                    # duplicating that feed in this terminal.
                    is_arm_related = (
                        'prearm' in low or 'not ready to arm' in low or
                        ('arm' in low and 'disarm' not in low) or
                        ('ekf3 imu' not in low and 'ekf' in low))
                    if is_arm_related:
                        with SS._lock:
                            SS.last_prearm_msg = text
                    self.get_logger().debug(f'FCU: {text}')
            except Exception as e:
                if 'no data' not in str(e):
                    self.get_logger().warn(f'Recv: {e}')
                time.sleep(0.05)

    def _pose_cb(self, msg):
        # Parse marker ID from frame_id (e.g. "aruco_id_0" or "aruco_id_1")
        frame_id = msg.header.frame_id
        try:
            msg_marker_id = int(frame_id.replace("aruco_id_", ""))
        except (ValueError, AttributeError):
            return  # malformed frame_id, ignore

        # Only accept if this matches our current target
        with SS._lock:
            target_id = SS.current_target_id
        if msg_marker_id != target_id:
            return  # different marker, ignore

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        cov = float(msg.pose.covariance[0])

        # Extract marker-relative yaw from the published quaternion.
        # aruco_pose_node.py builds this as a pure Z-axis rotation via
        # Rotation.from_euler("z", yaw).as_quat(), so for qx=qy=0 the
        # yaw angle recovers exactly as 2*atan2(qz, qw).
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)
        marker_yaw = 2.0 * math.atan2(qz, qw)
        # Wrap to [-pi, pi]
        marker_yaw = math.atan2(math.sin(marker_yaw), math.cos(marker_yaw))

        with SS._lock:
            SS.pose_x   = x
            SS.pose_y   = y
            SS.pose_cov = cov
            SS.pose_t   = time.time()
            SS.marker_yaw   = marker_yaw
            SS.marker_yaw_t = time.time()
            if cov < COV_MAX_ACCEPT: SS.confirmed += 1
            else: SS.confirmed = 0

    def _input_loop(self):
        self.get_logger().info('Commands: arm | forcearm | land | abort | status')
        while True:
            try:
                cmd = input().strip().lower()
                if cmd == 'arm':
                    if SS.state == STATE_IDLE:
                        with SS._lock:
                            SS.arm_mode = 'normal'
                        self.get_logger().info(
                            'Arming (normal — FCU prearm checks apply)')
                        SS.set_state(STATE_ARMING, self.get_logger())
                    else:
                        self.get_logger().warn(
                            f'Already {STATE_NAMES[SS.state]}')
                elif cmd == 'forcearm':
                    if SS.state == STATE_IDLE:
                        with SS._lock:
                            SS.arm_mode = 'force'
                        self.get_logger().warn(
                            'FORCE ARMING — bypassing FCU prearm checks. '
                            'Confirm you understand and accept any active '
                            'fault before this proceeds.')
                        SS.set_state(STATE_ARMING, self.get_logger())
                    else:
                        self.get_logger().warn(
                            f'Already {STATE_NAMES[SS.state]}')
                elif cmd == 'land':
                    SS.set_state(STATE_LANDING, self.get_logger())
                elif cmd == 'abort':
                    self.get_logger().warn('ABORT')
                    if self.mav and self.connected: mav_land(self.mav)
                    SS.set_state(STATE_DONE, self.get_logger())
                elif cmd == 'status':
                    with SS._lock:
                        print(f'State:{STATE_NAMES[SS.state]} '
                              f'Armed:{SS.armed} '
                              f'ArmMode:{SS.arm_mode} '
                              f'Alt:{SS.alt_m:.3f}m '
                              f'Pos:({SS.pos_x:.3f},{SS.pos_y:.3f}) '
                              f'Pose:({SS.pose_x:.3f},{SS.pose_y:.3f}) '
                              f'cov:{SS.pose_cov:.4f}')
                        if SS.last_prearm_msg:
                            print(f'  Last FCU prearm/EKF msg: '
                                  f'{SS.last_prearm_msg}')
            except EOFError: break
            except Exception as e:
                self.get_logger().error(f'Input: {e}')

    def _heading_fix_done(self, pos_x, pos_y):
        """
        Called once heading-fix completes (aligned or timed out).
        Re-enters STATE_CENTERING for a full second lock+hold+RMSE
        pass on the SAME marker before driving to the next one — yaw
        rotation during heading-fix can leave residual position drift,
        and this second pass closes that out properly (not just a
        quick re-converge) before Phase C/D's drive. current_target_id
        is deliberately NOT switched yet, so this pass keeps tracking
        whichever marker we just aligned to. next_phase_c remembers
        which NAV state to enter once THIS second pass completes —
        the routing happens in the hold-complete branch of
        STATE_CENTERING, distinguished via phase_b_complete /
        phase_c_complete already being True at that point (set when
        we first entered heading-fix, below).
        """
        with SS._lock:
            phase_c_done = SS.phase_c_complete
            SS.next_phase_c = not phase_c_done
            # Reset centering/lock state for the fresh pass — same
            # reset block used everywhere else a new centering pass
            # begins.
            SS.last_tgt_n = None
            SS.last_tgt_e = None
            SS.lock_start_t = None
            SS.rmse_samples = []
            SS.lock_anchor_n = None
            SS.lock_anchor_e = None
            SS.confirmed = 0
            SS.recovery_anchor_n = None
            SS.recovery_anchor_e = None
            SS.recovery_pending = False

        self.get_logger().info(
            f'→ Re-centering (full pass) on marker '
            f'{"A" if not phase_c_done else "B"} before driving on')
        SS.set_state(STATE_CENTERING, self.get_logger())

    def _send_loop(self):
        rate = 1.0/15.0
        while not self.connected: time.sleep(0.1)
        self.get_logger().info('Send loop active')

        while True:
            t0 = time.time()
            if not self.connected:
                time.sleep(rate); continue
            try:
                with SS._lock:
                    state      = SS.state
                    armed      = SS.armed
                    alt_m      = SS.alt_m
                    ground_alt = SS.ground_alt
                    t_sent     = SS.takeoff_sent
                    pos_x      = SS.pos_x
                    pos_y      = SS.pos_y
                    pos_z      = SS.pos_z
                    x          = SS.pose_x
                    y          = SS.pose_y
                    cov        = SS.pose_cov
                    conf       = SS.confirmed
                    pose_t     = SS.pose_t
                    reacquire  = SS.reacquire
                    arm_mode   = SS.arm_mode
                    marker_yaw   = SS.marker_yaw
                    marker_yaw_t = SS.marker_yaw_t

                dist = math.sqrt(x**2 + y**2)
                gained = alt_m - ground_alt
                detected = (cov < COV_MAX_ACCEPT and
                            conf >= CONFIRM_FRAMES and
                            pose_t is not None and
                            time.time() - pose_t < 2.0)

                # Centering yaw-rate correction: same proportional logic
                # as STATE_HEADING_FIX, but running continuously every
                # cycle during STATE_CENTERING rather than as a one-time
                # alignment, so the drone stays square to the marker
                # throughout the approach rather than only at the
                # Phase B->C handoff. Computed once here, used by both
                # the active-approach and anchor-hold mav_velocity_ned
                # calls below. Falls back to 0.0 (no correction, same
                # as locking heading) if marker_yaw is missing or stale
                # — never commands a correction from outdated data.
                centering_yaw_rate = 0.0
                myaw_fresh = (marker_yaw is not None and
                              marker_yaw_t is not None and
                              time.time() - marker_yaw_t < HEADING_FIX_STALE_S)
                if myaw_fresh and abs(math.degrees(marker_yaw)) > HEADING_FIX_TOLERANCE_DEG:
                    centering_yaw_rate = max(-HEADING_FIX_MAX_RATE,
                        min(HEADING_FIX_MAX_RATE, HEADING_FIX_K * marker_yaw))

                # Estimate drone velocity (NED) from consecutive position
                # samples — used to detect residual momentum entering the
                # near zone, so we can actively brake instead of relying
                # solely on mav_hold() to arrest drift via the FCU's own
                # loiter controller (which is comparatively slow).
                now_t = time.time()
                with SS._lock:
                    prev_n = SS.prev_pos_n
                    prev_e = SS.prev_pos_e
                    prev_t = SS.prev_pos_t
                if prev_n is not None and prev_t is not None:
                    dt = now_t - prev_t
                    if dt > 0.01:  # avoid div-by-near-zero on fast cycles
                        with SS._lock:
                            SS.est_vel_n = (pos_x - prev_n) / dt
                            SS.est_vel_e = (pos_y - prev_e) / dt
                with SS._lock:
                    SS.prev_pos_n = pos_x
                    SS.prev_pos_e = pos_y
                    SS.prev_pos_t = now_t
                    est_vel_n = SS.est_vel_n
                    est_vel_e = SS.est_vel_e

                if state == STATE_IDLE:
                    pass

                elif state == STATE_ARMING:
                    force = (arm_mode == 'force')
                    mav_arm(self.mav, force=force)
                    elapsed = SS.time_in_state()
                    if armed:
                        self.get_logger().info(
                            f'Armed ✅ (mode={arm_mode})')
                        with SS._lock:
                            SS.ground_alt   = alt_m
                            SS.takeoff_sent = False
                        self.get_logger().info(f'Ground ref: {alt_m:.3f}m')
                        SS.set_state(STATE_ARMED_WAIT, self.get_logger())
                    elif elapsed > 15.0:
                        with SS._lock:
                            last_msg = SS.last_prearm_msg
                        self.get_logger().error(
                            'Arm timeout — FCU never confirmed armed. '
                            f'Last prearm/EKF msg: {last_msg}')
                        SS.set_state(STATE_IDLE, self.get_logger())

                elif state == STATE_ARMED_WAIT:
                    elapsed = SS.time_in_state()
                    if elapsed < 3.0:
                        if int(elapsed*10) % 5 == 0:
                            self.get_logger().info(
                                f'Takeoff in {3.0-elapsed:.1f}s...')
                    else:
                        SS.set_state(STATE_TAKEOFF, self.get_logger())

                elif state == STATE_TAKEOFF:
                    if gained >= 0.75:
                        mav_hold(self.mav)
                        self.get_logger().info(
                            f'Altitude reached ✅ gained={gained:.2f}m')
                        SS.set_state(STATE_HOVER, self.get_logger())
                    elif SS.time_in_state() > 4.0 and gained < 0.3:
                        self.get_logger().warn(
                            f'Time fallback — alt={alt_m:.3f}m')
                        SS.set_state(STATE_HOVER, self.get_logger())
                    elif SS.time_in_state() > 20.0:
                        self.get_logger().error('Takeoff timeout')
                        SS.set_state(STATE_LANDING, self.get_logger())
                    elif not t_sent:
                        mav_takeoff(self.mav, TAKEOFF_ALT_M)
                        with SS._lock: SS.takeoff_sent = True
                        self.get_logger().info(
                            f'Takeoff cmd sent → {TAKEOFF_ALT_M}m')
                        self.mav.mav.request_data_stream_send(
                            self.mav.target_system,
                            self.mav.target_component,
                            mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                            10, 1)
                    else:
                        with SS._lock:
                            last_t = SS.alt_last_t
                        if time.time() - last_t > 1.5:
                            self.mav.mav.request_data_stream_send(
                                self.mav.target_system,
                                self.mav.target_component,
                                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                                10, 1)
                        self.get_logger().info(
                            f'Climbing... alt={alt_m:.3f}m '
                            f'gained={gained:.3f}m',
                            throttle_duration_sec=0.5)

                elif state == STATE_HOVER:
                    mav_hold(self.mav)
                    elapsed = SS.time_in_state()
                    wait = 0.3 if reacquire else HOVER_STABLE_S
                    if elapsed < wait:
                        self.get_logger().info(
                            f'Stabilising at {alt_m:.2f}m — '
                            f'{wait-elapsed:.1f}s...',
                            throttle_duration_sec=0.5)
                    else:
                        if detected:
                            self.get_logger().info(
                                f'Marker detected — '
                                f'x={x:.3f}m y={y:.3f}m '
                                f'dist={dist:.3f}m')
                            with SS._lock:
                                was_reacquire = SS.reacquire
                                SS.reacquire = False
                                SS.last_tgt_n = None
                                SS.last_tgt_e = None
                                # If this is a re-acquisition AND we have
                                # a recovery anchor, fly back to it first
                                if was_reacquire and SS.recovery_anchor_n is not None:
                                    SS.recovery_pending = True
                                    SS.recovery_start_t = time.time()
                                    self.get_logger().info(
                                        f'↩ Recovery armed: will fly to '
                                        f'({SS.recovery_anchor_n:.3f}, '
                                        f'{SS.recovery_anchor_e:.3f}) first')
                            SS.set_state(STATE_CENTERING,
                                         self.get_logger())
                        else:
                            self.get_logger().info(
                                'Waiting for marker...',
                                throttle_duration_sec=1.0)

                elif state == STATE_CENTERING:
                    if not detected:
                        self.get_logger().warn('Marker lost → hover')
                        with SS._lock:
                            SS.reacquire = True
                            SS.last_tgt_n = None
                            SS.last_tgt_e = None
                            SS.lock_start_t = None
                            SS.rmse_samples = []
                            SS.lock_anchor_n = None
                            SS.lock_anchor_e = None
                            SS.recovery_pending = False
                            SS.was_in_near_zone = False
                            SS.braking_active = False
                            SS.in_active_servo = False
                            SS.marker_repeat_count = 0
                            SS.prev_marker_x = None
                            SS.prev_marker_y = None
                        SS.set_state(STATE_HOVER, self.get_logger())
                    else:
                        # Update recovery anchor whenever marker is FRESH
                        # (cov < 0.05 = aruco_pose reports good detection, not stale)
                        if cov < 0.05:
                            with SS._lock:
                                SS.recovery_anchor_n = pos_x
                                SS.recovery_anchor_e = pos_y

                        # Handle one-shot recovery: fly to recovery anchor first
                        with SS._lock:
                            rec_pending = SS.recovery_pending
                            rec_n = SS.recovery_anchor_n
                            rec_e = SS.recovery_anchor_e
                            rec_start = SS.recovery_start_t
                        if rec_pending and rec_n is not None:
                            # Check completion: at anchor (within 5cm) OR timeout (2s)
                            dist_to_rec = math.sqrt(
                                (pos_x - rec_n)**2 + (pos_y - rec_e)**2)
                            elapsed_rec = time.time() - rec_start
                            if dist_to_rec < 0.05 or elapsed_rec > 2.0:
                                with SS._lock:
                                    SS.recovery_pending = False
                                self.get_logger().info(
                                    f'↩ Recovery complete: '
                                    f'dist_to_anchor={dist_to_rec:.3f}m '
                                    f'elapsed={elapsed_rec:.1f}s')
                            else:
                                # Send position command to recovery anchor
                                mav_goto_ned(self.mav, rec_n, rec_e,
                                             -TAKEOFF_ALT_M)
                                self.get_logger().info(
                                    f'↩ Recovery: flying to '
                                    f'({rec_n:.3f}, {rec_e:.3f}) | '
                                    f'dist={dist_to_rec:.3f}m '
                                    f't={elapsed_rec:.1f}s',
                                    throttle_duration_sec=0.5)
                                # Skip normal centering this cycle
                                sleep_t = rate - (time.time() - t0)
                                if sleep_t > 0: time.sleep(sleep_t)
                                continue

                        # Three-zone control with hysteresis on the
                        # near-zone <-> active-servo boundary:
                        # enter near-zone hold below NEAR_ZONE_ENTER_M,
                        # but only resume active servoing once dist
                        # exceeds NEAR_ZONE_EXIT_M (a higher threshold) —
                        # prevents a single noisy sample from flip-
                        # flopping between hold and active translation.
                        with SS._lock:
                            was_active_servo = SS.in_active_servo

                        if was_active_servo:
                            # Currently servoing — only drop back to
                            # near-zone hold once clearly back inside
                            # (use the tighter ENTER threshold)
                            should_hold = (dist < NEAR_ZONE_ENTER_M and
                                           SS.lock_anchor_n is None)
                        else:
                            # Currently holding — only leave hold once
                            # clearly outside (use the looser EXIT
                            # threshold, requires more margin to re-engage)
                            should_hold = (dist < NEAR_ZONE_EXIT_M and
                                           SS.lock_anchor_n is None)

                        with SS._lock:
                            SS.in_active_servo = not should_hold

                        if should_hold:
                            speed = math.sqrt(est_vel_n**2 + est_vel_e**2)

                            # Detect fresh entry into the near zone
                            with SS._lock:
                                just_entered = not SS.was_in_near_zone
                                SS.was_in_near_zone = True

                            BRAKE_SPEED_THRESHOLD = 0.05   # m/s — below this,
                                                            # not worth braking
                            BRAKE_GAIN = 0.6                # fraction of
                                                            # est. velocity to
                                                            # counter-command
                            BRAKE_DURATION_S = 0.3          # one-shot pulse

                            if just_entered and speed > BRAKE_SPEED_THRESHOLD:
                                with SS._lock:
                                    SS.braking_active = True
                                    SS.braking_start_t = time.time()
                                self.get_logger().info(
                                    f'NEAR-CENTER brake: est_vel='
                                    f'({est_vel_n:+.3f},{est_vel_e:+.3f})m/s '
                                    f'speed={speed:.3f}m/s — '
                                    f'applying counter-velocity pulse')

                            with SS._lock:
                                braking = SS.braking_active
                                brake_start = SS.braking_start_t

                            if braking and (time.time() - brake_start
                                             < BRAKE_DURATION_S):
                                # Command velocity opposed to estimated
                                # drift direction, scaled by BRAKE_GAIN
                                mav_velocity_ned(
                                    self.mav,
                                    -est_vel_n * BRAKE_GAIN,
                                    -est_vel_e * BRAKE_GAIN,
                                    0.0)
                                self.get_logger().info(
                                    f'NEAR-CENTER braking: x={x:+.3f} '
                                    f'y={y:+.3f} dist={dist:.3f} '
                                    f'vBrake=({-est_vel_n*BRAKE_GAIN:+.3f},'
                                    f'{-est_vel_e*BRAKE_GAIN:+.3f})m/s',
                                    throttle_duration_sec=0.2)
                            else:
                                if braking:
                                    with SS._lock:
                                        SS.braking_active = False
                                # Settled (or braking pulse complete) —
                                # zero-velocity hold as before
                                mav_hold(self.mav)
                                self.get_logger().info(
                                    f'NEAR-CENTER hold: x={x:+.3f} '
                                    f'y={y:+.3f} dist={dist:.3f}',
                                    throttle_duration_sec=1.0)
                            # Skip the position command logic below
                            # but still track lock time
                            tgt_n = pos_x  # for logging only
                            tgt_e = pos_y
                            tgt_d = -TAKEOFF_ALT_M
                            skip_position_cmd = True
                            velocity_servo_active = False
                        else:
                            with SS._lock:
                                SS.was_in_near_zone = False
                                SS.braking_active = False
                            # Active servoing — VELOCITY control, not
                            # absolute position target. x/y (marker offset)
                            # come directly from the camera, independent of
                            # EKF pos_x/pos_y, so commanding velocity here
                            # avoids compounding EKF position drift into
                            # the target over many cycles the way
                            # tgt_n = pos_x + step_n previously did.
                            #
                            # STALENESS GUARD: a velocity command keeps
                            # the drone physically translating every
                            # cycle it's resent, unlike a position target.
                            # pose_t alone is unreliable here because
                            # aruco_pose_node.py republishes the LAST
                            # KNOWN pose on a steady schedule even when
                            # detection has gone stale (ramping covariance
                            # instead of changing the timestamp pattern),
                            # so a timestamp-only check still reads
                            # "fresh" on repeated stale data — confirmed
                            # via flight log showing x/y frozen for ~0.9s
                            # while pose_t kept updating normally. We
                            # instead check covariance directly AND
                            # detect bit-identical repeats of x/y across
                            # consecutive cycles as a second, independent
                            # signal.
                            with SS._lock:
                                if (SS.prev_marker_x == x and
                                        SS.prev_marker_y == y):
                                    SS.marker_repeat_count += 1
                                else:
                                    SS.marker_repeat_count = 0
                                SS.prev_marker_x = x
                                SS.prev_marker_y = y
                                repeat_count = SS.marker_repeat_count

                            is_stale = (cov > CENTER_VEL_COV_MAX or
                                        repeat_count > CENTER_VEL_MAX_REPEATS)

                            if is_stale:
                                mav_hold(self.mav)
                                self.get_logger().info(
                                    f'Centering: stale pose '
                                    f'(cov={cov:.3f} repeats={repeat_count}) '
                                    f'— holding instead of servoing on '
                                    f'outdated x/y',
                                    throttle_duration_sec=0.3)
                                tgt_n = pos_x
                                tgt_e = pos_y
                                tgt_d = -TAKEOFF_ALT_M
                                skip_position_cmd = True
                                velocity_servo_active = False
                            else:
                                k_vel = (CENTER_VEL_K_FAR if dist > 0.15
                                         else CENTER_VEL_K_NEAR)
                                # AXIS SIGN FIX (confirmed via bench test
                                # with drone held stationary, marker
                                # moved to known physical positions):
                                # x: marker LOW in frame (x>0) = marker
                                #    BEHIND drone -> need NEGATIVE N to
                                #    close gap. Marker HIGH in frame
                                #    (x<0) = marker AHEAD -> need
                                #    POSITIVE N. So vn = -x * k, not +x*k.
                                # y: marker on camera-LEFT (y>0) = with
                                #    drone facing ~north, camera-left is
                                #    NED-WEST -> need NEGATIVE E to close
                                #    gap. Marker on camera-RIGHT (y<0) =
                                #    NED-EAST -> need POSITIVE E.
                                #    So ve = -y * k, not +y*k.
                                # Both axes were previously using the raw
                                # camera value directly as the NED
                                # command with no sign flip, driving the
                                # drone AWAY from the marker on both
                                # axes — very likely the root cause of
                                # the persistent directional drift seen
                                # across nearly every flight log this
                                # session.
                                vn = max(-CENTER_VEL_MAX,
                                         min(CENTER_VEL_MAX, -x * k_vel))
                                ve = max(-CENTER_VEL_MAX,
                                         min(CENTER_VEL_MAX, -y * k_vel))
                                mav_velocity_ned(self.mav, vn, ve, 0.0,
                                                  centering_yaw_rate)
                                # tgt_n/tgt_e kept only for logging
                                # consistency with the position-based
                                # locked/anchor branch
                                tgt_n = pos_x + vn * rate
                                tgt_e = pos_y + ve * rate
                                tgt_d = -TAKEOFF_ALT_M
                                skip_position_cmd = True
                                velocity_servo_active = True

                        # Anchor hold with 3-tier behaviour:
                        #   dist < DEADBAND_M: rigid hold at fixed anchor
                        #   DEADBAND_M < dist < ANCHOR_RELEASE_M: gently shift anchor
                        #   dist >= ANCHOR_RELEASE_M: release anchor
                        #
                        # CONVERTED TO VELOCITY CONTROL: mav_goto_ned
                        # sends an absolute 3D position target (N, E, AND
                        # D together) every time it fires. Even though D
                        # was always the same constant (-TAKEOFF_ALT_M),
                        # repeatedly re-asserting an absolute Z position
                        # target appears to have been triggering real
                        # altitude disturbances (~0.8m climb-then-drop
                        # confirmed via CTUN+RFND cross-check, timed
                        # right around anchor formation/hold) — likely
                        # interacting with the still-unfixed EK3_SRC
                        # Baro Z-source instability every time a fresh
                        # position setpoint message arrives. Velocity
                        # control with vd=0.0 only asserts "don't move
                        # vertically," never re-asserting an absolute Z,
                        # so it shouldn't retrigger this disturbance —
                        # same fix rationale as the active-approach
                        # branch, now applied to anchor-hold too.
                        ANCHOR_HOLD_K = 1.5  # m/s per meter of N/E error
                                              # from anchor point — firm,
                                              # since this is a short-
                                              # range correction back to
                                              # a known-good fixed point,
                                              # not chasing noisy live
                                              # camera offset
                        with SS._lock:
                            anchor_n = SS.lock_anchor_n
                            anchor_e = SS.lock_anchor_e

                        if anchor_n is not None:
                            if dist < DEADBAND_M:
                                # Rigid anchor hold — velocity toward the
                                # fixed anchor point, computed from
                                # current EKF position vs anchor
                                err_n = anchor_n - pos_x
                                err_e = anchor_e - pos_y
                                use_anchor = True
                            elif dist < ANCHOR_RELEASE_M:
                                # Drift zone: gently shift anchor toward marker
                                # Soft correction with very small gain
                                # Same axis sign fix as the active-servo
                                # branch above: -x for N, -y for E.
                                ANCHOR_DRIFT_K = 0.02
                                with SS._lock:
                                    SS.lock_anchor_n = anchor_n + (-x) * ANCHOR_DRIFT_K
                                    SS.lock_anchor_e = anchor_e + (-y) * ANCHOR_DRIFT_K
                                    anchor_n = SS.lock_anchor_n
                                    anchor_e = SS.lock_anchor_e
                                err_n = anchor_n - pos_x
                                err_e = anchor_e - pos_y
                                use_anchor = True
                            else:
                                # Beyond release threshold — drop anchor
                                with SS._lock:
                                    SS.lock_anchor_n = None
                                    SS.lock_anchor_e = None
                                self.get_logger().info(
                                    'Anchor released — marker drifted')
                                use_anchor = False
                        else:
                            use_anchor = False

                        if use_anchor:
                            anchor_vn = max(-CENTER_VEL_MAX,
                                min(CENTER_VEL_MAX, err_n * ANCHOR_HOLD_K))
                            anchor_ve = max(-CENTER_VEL_MAX,
                                min(CENTER_VEL_MAX, err_e * ANCHOR_HOLD_K))
                            mav_velocity_ned(self.mav, anchor_vn,
                                              anchor_ve, 0.0,
                                              centering_yaw_rate)
                            tgt_n = pos_x + anchor_vn * rate  # logging only
                            tgt_e = pos_y + anchor_ve * rate
                            tgt_d = -TAKEOFF_ALT_M
                            skip_position_cmd = True


                        # Track lock time and set anchor when stable
                        with SS._lock:
                            if dist < DEADBAND_M:
                                if SS.lock_start_t is None:
                                    SS.lock_start_t = time.time()
                                    SS.rmse_samples = []
                                    self.get_logger().info(
                                        f'LOCKED ✅ dist={dist:.3f}m '
                                        f'x={x:.3f}m y={y:.3f}m')
                                SS.rmse_samples.append((x, y))
                                elapsed_lock = time.time() - SS.lock_start_t
                                n = len(SS.rmse_samples)
                                # After 2s stable, anchor current position
                                if (SS.lock_anchor_n is None and
                                    elapsed_lock > ANCHOR_DELAY_S):
                                    SS.lock_anchor_n = pos_x
                                    SS.lock_anchor_e = pos_y
                                    self.get_logger().info(
                                        f'⚓ Anchor set: '
                                        f'({pos_x:.3f}, {pos_y:.3f}) '
                                        f'— stable hold engaged')
                            else:
                                elapsed_lock = 0 if SS.lock_start_t is None \
                                    else time.time() - SS.lock_start_t
                                n = len(SS.rmse_samples)

                        # Log differently based on lock state
                        if dist < DEADBAND_M:
                            mode = 'ANCHOR' if use_anchor else 'HOLD'
                            self.get_logger().info(
                                f'{mode}: x={x:+.3f}m y={y:+.3f}m '
                                f't={elapsed_lock:.1f}s/{HOLD_DURATION_S:.0f}s '
                                f'n={n}',
                                throttle_duration_sec=1.0)
                        else:
                            self.get_logger().info(
                                f'Centering: x={x:+.3f}m y={y:+.3f}m '
                                f'dist={dist:.3f}m | '
                                f'→N={tgt_n:+.3f} E={tgt_e:+.3f}m')

                        # Complete after HOLD_DURATION_S of locked time
                        if elapsed_lock >= HOLD_DURATION_S:
                            self.get_logger().info(
                                f'Hold complete — {n} samples')
                            with SS._lock:
                                samples = list(SS.rmse_samples)
                                phase_b_done = SS.phase_b_complete
                                phase_c_done = SS.phase_c_complete
                                cur_target = SS.current_target_id
                                next_phase_c = SS.next_phase_c
                            save_rmse(samples, self.get_logger())

                            # Three-way decision based on which marker
                            # we just finished centering/holding on:
                            # target 0 (marker A) -> heading-fix -> Phase C
                            # target 1 (marker B) -> heading-fix -> Phase D
                            # target 2 (marker C) -> done, land
                            # The shared reset block (recovery anchor,
                            # centering state, yaw repeat tracking) is
                            # identical for both heading-fix entries —
                            # only the log message and which "done" flag
                            # gets set differs.
                            # Four-way decision based on which marker
                            # we just finished centering/holding on,
                            # and whether this was the FIRST pass on
                            # that marker (-> go do heading-fix) or the
                            # SECOND pass, run after heading-fix
                            # already completed (-> drive to the next
                            # marker now). next_phase_c is None until
                            # _heading_fix_done() sets it, so its
                            # presence is exactly the "second pass"
                            # signal.
                            if next_phase_c is not None:
                                # Second pass complete (post-heading-fix
                                # re-centering on the same marker) —
                                # NOW switch target and start the drive.
                                with SS._lock:
                                    SS.nav_start_t = time.time()
                                    SS.nav_start_pos_n = pos_x
                                    SS.nav_start_pos_e = pos_y
                                    SS.next_phase_c = None  # consumed
                                    if next_phase_c:
                                        SS.current_target_id = 1
                                    else:
                                        SS.current_target_id = 2
                                if next_phase_c:
                                    self.get_logger().info(
                                        f'→ Phase C: open-loop velocity '
                                        f'nav, {NAV_VEL_MPS:.2f}m/s north '
                                        f'for {NAV_NORTH_S:.2f}s '
                                        f'(~{NAV_DISTANCE_M}m) for marker '
                                        f'B (ID 1)')
                                    SS.set_state(STATE_NAVIGATE_TO_B,
                                                  self.get_logger())
                                else:
                                    self.get_logger().info(
                                        f'→ Phase D: open-loop velocity '
                                        f'nav, {NAV_VEL_MPS:.2f}m/s north '
                                        f'for {NAV_NORTH_S:.2f}s '
                                        f'(~{NAV_DISTANCE_M}m) for marker '
                                        f'C (ID 2)')
                                    SS.set_state(STATE_NAVIGATE_TO_C,
                                                  self.get_logger())
                            elif not phase_b_done and cur_target == 0:
                                # Just finished Phase B on marker A.
                                # Do a one-time heading alignment to marker A
                                # BEFORE switching target / driving forward —
                                # current_target_id stays 0 so marker_yaw
                                # keeps updating from marker A during the fix.
                                with SS._lock:
                                    SS.phase_b_complete = True
                                    SS.heading_fix_start_t = time.time()
                                    SS.prev_marker_yaw = None
                                    SS.marker_yaw_repeat_count = 0
                                    # Reset recovery anchor (A's anchor not useful for B)
                                    SS.recovery_anchor_n = None
                                    SS.recovery_anchor_e = None
                                    SS.recovery_pending = False
                                    # Reset centering state
                                    SS.last_tgt_n = None
                                    SS.last_tgt_e = None
                                    SS.lock_start_t = None
                                    SS.rmse_samples = []
                                    SS.lock_anchor_n = None
                                    SS.lock_anchor_e = None
                                    SS.confirmed = 0
                                self.get_logger().info(
                                    '→ Heading fix: aligning to marker A '
                                    f'before Phase C (tolerance '
                                    f'{HEADING_FIX_TOLERANCE_DEG:.1f}°)')
                                SS.set_state(STATE_HEADING_FIX,
                                             self.get_logger())
                            elif not phase_c_done and cur_target == 1:
                                # Just finished centering on marker B.
                                # Same one-time heading alignment, this
                                # time to marker B, before Phase D's
                                # drive toward marker C (ID 2).
                                with SS._lock:
                                    SS.phase_c_complete = True
                                    SS.heading_fix_start_t = time.time()
                                    SS.prev_marker_yaw = None
                                    SS.marker_yaw_repeat_count = 0
                                    # Reset recovery anchor (B's anchor not useful for C)
                                    SS.recovery_anchor_n = None
                                    SS.recovery_anchor_e = None
                                    SS.recovery_pending = False
                                    # Reset centering state
                                    SS.last_tgt_n = None
                                    SS.last_tgt_e = None
                                    SS.lock_start_t = None
                                    SS.rmse_samples = []
                                    SS.lock_anchor_n = None
                                    SS.lock_anchor_e = None
                                    SS.confirmed = 0
                                self.get_logger().info(
                                    '→ Heading fix: aligning to marker B '
                                    f'before Phase D (tolerance '
                                    f'{HEADING_FIX_TOLERANCE_DEG:.1f}°)')
                                SS.set_state(STATE_HEADING_FIX,
                                             self.get_logger())
                            else:
                                # Finished marker C (final target), or
                                # unexpected state — land
                                SS.set_state(STATE_LANDING,
                                             self.get_logger())

                elif state == STATE_HEADING_FIX:
                    # One-time yaw alignment to marker A before driving
                    # forward toward marker B. Uses ArUco rvec-derived yaw
                    # (ground-truth relative to the marker), NOT gyro —
                    # only available while marker A is confidently in view.
                    with SS._lock:
                        myaw   = SS.marker_yaw
                        myaw_t = SS.marker_yaw_t
                        fix_start = SS.heading_fix_start_t
                        if SS.prev_marker_yaw == myaw:
                            SS.marker_yaw_repeat_count += 1
                        else:
                            SS.marker_yaw_repeat_count = 0
                        SS.prev_marker_yaw = myaw
                        yaw_repeat_count = SS.marker_yaw_repeat_count

                    elapsed = time.time() - fix_start
                    # STALENESS: same covariance + repeat-count check as
                    # STATE_CENTERING's CENTER_VEL_STALE check — a
                    # timestamp-only freshness check misses
                    # aruco_pose_node.py's stale-republish behavior
                    # (same value, fresh timestamp, rising covariance).
                    # Confirmed via flight log: heading-fix error froze
                    # at -18.36deg for ~2.3s while myaw_t kept looking
                    # fresh under the old timestamp-only check.
                    myaw_fresh = (myaw is not None and myaw_t is not None
                                  and time.time() - myaw_t < HEADING_FIX_STALE_S
                                  and cov <= CENTER_VEL_COV_MAX
                                  and yaw_repeat_count <= CENTER_VEL_MAX_REPEATS)

                    if elapsed > HEADING_FIX_TIMEOUT_S:
                        self.get_logger().warn(
                            f'Heading-fix timeout ({elapsed:.1f}s) — '
                            f'proceeding to Phase C regardless')
                        mav_hold(self.mav)
                        self._heading_fix_done(pos_x, pos_y)
                    elif not myaw_fresh:
                        # Marker not confidently visible — hold and wait,
                        # bounded by the same overall timeout above
                        mav_hold(self.mav)
                        self.get_logger().info(
                            f'Heading-fix: waiting for fresh marker yaw '
                            f'(cov={cov:.3f} repeats={yaw_repeat_count})...',
                            throttle_duration_sec=0.5)
                    else:
                        yaw_err_deg = math.degrees(myaw)
                        if abs(yaw_err_deg) <= HEADING_FIX_TOLERANCE_DEG:
                            self.get_logger().info(
                                f'✓ Heading aligned: {yaw_err_deg:+.2f}° '
                                f'(within {HEADING_FIX_TOLERANCE_DEG:.1f}°) '
                                f'after {elapsed:.1f}s')
                            self._heading_fix_done(pos_x, pos_y)
                        else:
                            # NOTE: marker_yaw is the marker's APPARENT
                            # rotation in camera frame, the inverse of the
                            # camera's (drone's) actual rotation — confirmed
                            # via heading_fix_test.py bench test. Correcting
                            # the drone's real heading needs a POSITIVE sign.
                            cmd_rate = HEADING_FIX_K * myaw
                            cmd_rate = max(-HEADING_FIX_MAX_RATE,
                                            min(HEADING_FIX_MAX_RATE, cmd_rate))
                            mav_yaw_rate(self.mav, cmd_rate)
                            self.get_logger().info(
                                f'Heading-fix: error={yaw_err_deg:+.2f}° '
                                f'cmd_rate={math.degrees(cmd_rate):+.1f}°/s '
                                f't={elapsed:.1f}s',
                                throttle_duration_sec=0.3)

                elif state == STATE_NAVIGATE_TO_B:
                    # Open-loop, timed velocity navigation — does NOT depend
                    # on LOCAL_POSITION_NED being healthy. Drives forward

                    # (north) at NAV_VEL_MPS for NAV_NORTH_S seconds, then
                    # brakes to zero velocity for NAV_RAMP_DOWN_S before
                    # declaring arrival. An absolute NAV_TIMEOUT_S safety
                    # cap still applies regardless of position data quality.
                    with SS._lock:
                        start_t = SS.nav_start_t
                        start_n = SS.nav_start_pos_n
                        start_e = SS.nav_start_pos_e

                    if start_t is None:
                        # Shouldn't happen, but safety fallback
                        self.get_logger().error('NAV: no start time set, landing')
                        SS.set_state(STATE_LANDING, self.get_logger())
                    else:
                        elapsed = time.time() - start_t

                        # Sanity-check distance traveled so far, using
                        # whatever position data is available — informational
                        # only, never gates the open-loop timing decision
                        if start_n is not None:
                            dist_traveled = math.sqrt(
                                (pos_x - start_n)**2 + (pos_y - start_e)**2)
                        else:
                            dist_traveled = float('nan')

                        if elapsed > NAV_TIMEOUT_S:
                            # Absolute safety cap — should never normally hit
                            # this since NAV_NORTH_S + NAV_RAMP_DOWN_S is
                            # well under NAV_TIMEOUT_S
                            mav_hold(self.mav)
                            self.get_logger().warn(
                                f'NAV timeout ({elapsed:.1f}s) — '
                                f'dist_traveled≈{dist_traveled:.3f}m — '
                                f'searching anyway')
                            SS.set_state(STATE_SEARCHING_B,
                                         self.get_logger())
                        elif elapsed < NAV_NORTH_S:
                            # Driving phase: constant forward velocity
                            mav_velocity_ned(self.mav, NAV_VEL_MPS, 0.0, 0.0)
                            self.get_logger().info(
                                f'NAV drive: t={elapsed:.2f}/{NAV_NORTH_S:.2f}s '
                                f'vN={NAV_VEL_MPS:.2f}m/s '
                                f'dist_traveled≈{dist_traveled:.3f}m',
                                throttle_duration_sec=0.5)
                        elif elapsed < NAV_NORTH_S + NAV_RAMP_DOWN_S:
                            # Braking phase: zero-velocity hold to kill
                            # residual momentum before searching for marker B
                            mav_hold(self.mav)
                            self.get_logger().info(
                                f'NAV brake: t={elapsed:.2f}s '
                                f'dist_traveled≈{dist_traveled:.3f}m',
                                throttle_duration_sec=0.5)
                        else:
                            # Drive + brake complete — declare arrival
                            self.get_logger().info(
                                f'✓ NAV complete after {elapsed:.2f}s '
                                f'(commanded ~{NAV_DISTANCE_M}m @ '
                                f'{NAV_VEL_MPS:.2f}m/s) — '
                                f'dist_traveled≈{dist_traveled:.3f}m')
                            SS.set_state(STATE_SEARCHING_B,
                                         self.get_logger())

                elif state == STATE_SEARCHING_B:
                    # Hold position, wait for marker B (ID 1) detection
                    mav_hold(self.mav)
                    elapsed = SS.time_in_state()

                    if detected:
                        # Marker B detected! Switch to CENTERING
                        self.get_logger().info(
                            f'✓ Marker B detected — '
                            f'x={x:.3f} y={y:.3f} dist={dist:.3f}m')
                        with SS._lock:
                            SS.last_tgt_n = None
                            SS.last_tgt_e = None
                            SS.reacquire = False
                        SS.set_state(STATE_CENTERING, self.get_logger())
                    elif elapsed > SEARCH_TIMEOUT_S:
                        self.get_logger().warn(
                            f'Search timeout ({elapsed:.1f}s) — '
                            f'marker B not found, landing at current position')
                        SS.set_state(STATE_LANDING, self.get_logger())
                    else:
                        self.get_logger().info(
                            f'Searching for marker B... '
                            f'{SEARCH_TIMEOUT_S - elapsed:.1f}s remaining',
                            throttle_duration_sec=1.0)

                elif state == STATE_NAVIGATE_TO_C:
                    # Open-loop, timed velocity navigation toward marker C
                    # (ID 2) — identical mechanism to STATE_NAVIGATE_TO_B,
                    # same NAV_DISTANCE_M/NAV_VEL_MPS, just a different
                    # destination state on completion/timeout.
                    with SS._lock:
                        start_t = SS.nav_start_t
                        start_n = SS.nav_start_pos_n
                        start_e = SS.nav_start_pos_e

                    if start_t is None:
                        self.get_logger().error('NAV: no start time set, landing')
                        SS.set_state(STATE_LANDING, self.get_logger())
                    else:
                        elapsed = time.time() - start_t

                        if start_n is not None:
                            dist_traveled = math.sqrt(
                                (pos_x - start_n)**2 + (pos_y - start_e)**2)
                        else:
                            dist_traveled = float('nan')

                        if elapsed > NAV_TIMEOUT_S:
                            mav_hold(self.mav)
                            self.get_logger().warn(
                                f'NAV timeout ({elapsed:.1f}s) — '
                                f'dist_traveled≈{dist_traveled:.3f}m — '
                                f'searching anyway')
                            SS.set_state(STATE_SEARCHING_C,
                                         self.get_logger())
                        elif elapsed < NAV_NORTH_S:
                            mav_velocity_ned(self.mav, NAV_VEL_MPS, 0.0, 0.0)
                            self.get_logger().info(
                                f'NAV drive: t={elapsed:.2f}/{NAV_NORTH_S:.2f}s '
                                f'vN={NAV_VEL_MPS:.2f}m/s '
                                f'dist_traveled≈{dist_traveled:.3f}m',
                                throttle_duration_sec=0.5)
                        elif elapsed < NAV_NORTH_S + NAV_RAMP_DOWN_S:
                            mav_hold(self.mav)
                            self.get_logger().info(
                                f'NAV brake: t={elapsed:.2f}s '
                                f'dist_traveled≈{dist_traveled:.3f}m',
                                throttle_duration_sec=0.5)
                        else:
                            self.get_logger().info(
                                f'✓ NAV complete after {elapsed:.2f}s '
                                f'(commanded ~{NAV_DISTANCE_M}m @ '
                                f'{NAV_VEL_MPS:.2f}m/s) — '
                                f'dist_traveled≈{dist_traveled:.3f}m')
                            SS.set_state(STATE_SEARCHING_C,
                                         self.get_logger())

                elif state == STATE_SEARCHING_C:
                    # Hold position, wait for marker C (ID 2) detection —
                    # identical mechanism to STATE_SEARCHING_B.
                    mav_hold(self.mav)
                    elapsed = SS.time_in_state()

                    if detected:
                        self.get_logger().info(
                            f'✓ Marker C detected — '
                            f'x={x:.3f} y={y:.3f} dist={dist:.3f}m')
                        with SS._lock:
                            SS.last_tgt_n = None
                            SS.last_tgt_e = None
                            SS.reacquire = False
                        SS.set_state(STATE_CENTERING, self.get_logger())
                    elif elapsed > SEARCH_TIMEOUT_S:
                        self.get_logger().warn(
                            f'Search timeout ({elapsed:.1f}s) — '
                            f'marker C not found, landing at current position')
                        SS.set_state(STATE_LANDING, self.get_logger())
                    else:
                        self.get_logger().info(
                            f'Searching for marker C... '
                            f'{SEARCH_TIMEOUT_S - elapsed:.1f}s remaining',
                            throttle_duration_sec=1.0)

                elif state == STATE_LOCKED:
                    # Deprecated — unified into CENTERING
                    SS.set_state(STATE_CENTERING, self.get_logger())

                elif state == STATE_LANDING:
                    # Only send Land mode command, NOT mav_hold
                    # (mav_hold sends velocity commands which conflict with Land mode)
                    mav_land(self.mav)
                    elapsed = SS.time_in_state()

                    # Track sustained disarm
                    if not hasattr(self, '_disarmed_since'):
                        self._disarmed_since = None
                    if not armed:
                        if self._disarmed_since is None:
                            self._disarmed_since = time.time()
                            self.get_logger().info(
                                f'Detected disarm at alt={alt_m:.3f}m')
                    else:
                        self._disarmed_since = None

                    # Require: sustained disarm (3s) AND low altitude (<0.15m)
                    # OR absolute timeout (30s)
                    sustained_disarm = (self._disarmed_since is not None and
                                        (time.time() - self._disarmed_since) > 3.0)

                    if sustained_disarm and alt_m < 0.15:
                        self.get_logger().info(
                            f'Landed ✅ alt={alt_m:.3f}m')
                        SS.set_state(STATE_DONE, self.get_logger())
                    elif elapsed > 10.0:
                        self.get_logger().warn(
                            f'Land timeout — forcing DONE alt={alt_m:.3f}m')
                        SS.set_state(STATE_DONE, self.get_logger())
                    else:
                        self.get_logger().info(
                            f'Landing... alt={alt_m:.3f}m '
                            f'armed={armed}',
                            throttle_duration_sec=1.0)

                elif state == STATE_DONE:
                    self.get_logger().info('Done. Ctrl+C to exit.')
                    break

            except Exception as e:
                self.get_logger().warn(f'Send: {e}')
                self.connected = False
                self._connect()

            sleep_t = rate - (time.time() - t0)
            if sleep_t > 0: time.sleep(sleep_t)


def main(args=None):
    rclpy.init(args=args)
    node = VisionSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
