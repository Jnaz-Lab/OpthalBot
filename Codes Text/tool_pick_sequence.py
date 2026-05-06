import cv2
import time
import json
import numpy as np
from enum import Enum
from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================

ROBOT_IP  = "192.168.0.100"
CAM_INDEX = 0        # 0 = first USB camera
USE_DSHOW = True     # DirectShow backend — more stable on Windows
DRY_RUN   = False    # True = print all robot commands without moving (safe testing)

ZERO_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Travel speeds — used when homing or moving between stations
JOINT_VEL_LIMIT       = 25
CART_LIN_VEL          = 400
CART_ACC              = 250

# Tracking speeds — used during visual servo loop (blending enabled for smooth motion)
TRACK_CART_LIN_VEL    = 200
TRACK_CART_ACC        = 400

# Approach speeds — used when moving to the search pose before tracking starts
APPROACH_CART_LIN_VEL = 150
APPROACH_CART_ACC     = 120

# Delays between steps of the tool sequence (seconds)
DELAY_AFTER_ZERO_S            = 1.0   # pause after homing
DELAY_AFTER_POSE_CAPTURE_S    = 2.0   # let the robot fully settle before saving the pose
DELAY_AFTER_PRINT_SUMMARY_S   = 2.0   # pause after printing the pose summary
DELAY_AFTER_OCT_AT_TARGET_S   = 1.0   # dwell time with OCT at target position
DELAY_AFTER_BIOPEN_AT_TARGET_S = 1.0  # dwell time with BioPen at target position

JOG_STEP_MM = 1.0    # how far each manual keypress moves the robot (mm)
MIN_Z_MM    = 100.0  # hard safety floor — robot will never descend below this Z

# Output files are saved next to this script
SAVE_DIR       = Path(__file__).parent / "pic"
POSE_FILE      = Path(__file__).parent / "pose_reference.json"


# =============================================================================
# OBJECT DETECTION (intensity-based, grayscale)
# =============================================================================

# Object is detected by finding pixels whose grayscale brightness is in [I_MIN, I_MAX].
I_MIN            = 30
I_MAX            = 100
MIN_CONTOUR_AREA = 200   # blobs smaller than this (px²) are ignored as noise

CAM_CENTER_RADIUS_PX = 10
CAM_CENTER_THICKNESS = 2


# =============================================================================
# SEARCH POSE — where the robot starts looking for the target
# =============================================================================

SEARCH_Z    = 250.0   # Z height (mm) when scanning
TRACK_ALPHA =   0.0
TRACK_BETA  =  90.0   # tool pointing straight down
TRACK_GAMMA =   0.0


# =============================================================================
# XY VISUAL SERVO CONTROL
# =============================================================================

DT         = 0.05    # control loop period (20 Hz)
AUTO_XY_ON = True    # False = disable XY servo (manual jog only)

# Deadband: pixel errors smaller than this are treated as zero.
# Larger deadband during Z approach avoids fighting the Z motion.
DEADBAND_PX          =  6.0
DEADBAND_PX_APPROACH = 10.0
CENTER_ACCEPT_PX     =  8.0   # XY must be within this (px) to count as aligned

# Low-pass filter coefficients (closer to 1 = faster response, closer to 0 = smoother)
ALPHA_CENTROID = 0.35
ALPHA_STEP_XY  = 0.46

GAIN_XY = 0.30   # proportional gain for XY corrections

# Pixel-to-mm conversion (calibrated with pixel_calibration.py).
# Diagonal matrix: converts [px_x, px_y] pixel error → [mm_x, mm_y] robot step.
MM_PER_PX_X  = 0.03
MM_PER_PX_Y  = 0.03
pixel_to_mm  = np.array([[MM_PER_PX_X, 0.0],
                          [0.0,         MM_PER_PX_Y]], dtype=float)

MAX_STEP_XY_MM       = 0.5     # max XY correction per loop cycle (mm)
FLIP_X               = False
FLIP_Y               = False
SWAP_XY              = True    # set True if camera X maps to robot Y (rotated mount)
XY_ALIGN_HOLD_FRAMES = 5       # frames XY must stay aligned before Z descent starts


# =============================================================================
# Z VISUAL SERVO CONTROL (area-based)
#
# The target fills more pixels as the robot descends.
# The robot descends until area_px reaches AREA_TARGET.
# Z only moves when XY is already aligned (XY takes priority).
# Press 't' at runtime to set AREA_TARGET from the current frame.
# =============================================================================

AUTO_Z_ON      = True    # False = disable Z servo (manual jog only)
AREA_TARGET    = 70000.0
AREA_TOLERANCE =  1000.0
FLIP_Z         = False

GAIN_Z_AREA   = 0.00012
ALPHA_STEP_Z  = 0.55
MAX_STEP_Z_MM = 0.8


# =============================================================================
# FINAL POSE AVERAGING
# When HOLD is reached, we average N pose/joint samples to get a stable
# reference. This reference is then used to drive the OCT and BioPen to the
# exact same location without relying on vision.
# =============================================================================

POSE_AVG_SAMPLES = 40
POSE_AVG_DELAY_S = 0.03   # seconds between each sample read
POSE_SETTLE_S    = 0.30   # time to wait for vibrations to die down before sampling


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    WAIT_DETECT = "WAIT_DETECT"
    ALIGN_XY    = "ALIGN_XY"
    APPROACH_Z  = "APPROACH_Z"
    HOLD        = "HOLD"

_STATE_COLORS = {
    State.ALIGN_XY:   (0, 255, 255),   # yellow
    State.APPROACH_Z: (0, 165, 255),   # orange
    State.HOLD:       (0, 255, 0),     # green
}


# =============================================================================
# ROBOT WRAPPER
# =============================================================================

class Meca500Client:
    def __init__(self, ip: str, dry_run: bool = True):
        self.ip         = ip
        self.dry_run    = dry_run
        self.robot      = None
        self._current_z = SEARCH_Z   # local Z tracker (avoids constant pose reads in servo loop)

    # ------------------------------------------------------------------
    # Internal helpers: parse raw API output into usable types
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pose(raw):
        # Mecademic API can return 6 or 7 values depending on firmware version.
        # 7-value form has a status byte at the start; we always want the last 6.
        if raw is None:
            return None
        values = list(raw)
        if len(values) >= 7:
            values = values[-6:]
        elif len(values) < 6:
            return None
        keys = ("x", "y", "z", "alpha", "beta", "gamma")
        return {k: float(v) for k, v in zip(keys, values)}

    @staticmethod
    def _parse_joints(raw):
        if raw is None:
            return None
        values = list(raw)
        if len(values) >= 7:
            values = values[-6:]
        elif len(values) < 6:
            return None
        return [float(v) for v in values]

    def _read_pose(self):
        # Try all known method names — different firmware versions use different names.
        if self.robot is None:
            return None
        for name in ("GetRtCartPos", "GetPose", "GetRtTargetCartPos"):
            try:
                val = getattr(self.robot, name)()
                if val is not None:
                    return val
            except Exception:
                pass
        return None

    def _read_joints(self):
        if self.robot is None:
            return None
        for name in ("GetRtJointPos", "GetRtTargetJointPos", "GetJoints"):
            try:
                val = getattr(self.robot, name)()
                if val is not None:
                    return val
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self.dry_run:
            print("[Robot] DRY_RUN — skipping connection.")
            return
        try:
            from mecademicpy.robot import Robot
        except Exception as e:
            raise RuntimeError("mecademicpy not found. Install it or set DRY_RUN=True.") from e

        self.robot = Robot()
        self.robot.Connect(self.ip)
        self.robot.ActivateRobot()
        self.robot.Home()
        try:
            self.robot.WaitHomed()
        except Exception:
            self.robot.WaitIdle()   # older firmware fallback
        self.robot.ResumeMotion()
        print("[Robot] Connected and homed.")

    # ------------------------------------------------------------------
    # Speed helpers
    # ------------------------------------------------------------------

    def _set_travel_speed(self):
        # Travel speed: used when moving between tool stations or homing.
        if self.dry_run or self.robot is None:
            return
        self.robot.SetJointVelLimit(JOINT_VEL_LIMIT)
        self.robot.SetCartLinVel(CART_LIN_VEL)
        self.robot.SetCartAcc(CART_ACC)
        try:
            self.robot.SetBlending(0)
        except Exception:
            pass
        self.robot.ResumeMotion()

    def _set_tracking_speed(self):
        # Tracking speed: used during the visual servo loop.
        # blending=100 makes the robot blend waypoints into one continuous trajectory
        # instead of stopping at each point, which keeps motion smooth during servo.
        if self.dry_run or self.robot is None:
            return
        self.robot.SetCartLinVel(TRACK_CART_LIN_VEL)
        self.robot.SetCartAcc(TRACK_CART_ACC)
        try:
            self.robot.SetBlending(100)
        except Exception:
            pass
        self.robot.ResumeMotion()
        print("[Robot] Tracking speed set (blending=100).")

    # ------------------------------------------------------------------
    # Basic motion
    # ------------------------------------------------------------------

    def move_zero_joints(self):
        if self.dry_run:
            print(f"[Robot] DRY_RUN MoveJoints{ZERO_JOINTS}")
            return
        self._set_travel_speed()
        self.robot.MoveJoints(*ZERO_JOINTS)
        self.robot.WaitIdle()
        print("[Robot] At zero joints.")

    def move_to_search_pose(self, z_mm: float):
        # Keep current XY and only change Z and orientation.
        if self.dry_run:
            print(f"[Robot] DRY_RUN → search pose Z={z_mm:.1f}")
            self._current_z = z_mm
            return
        raw  = self._read_pose()
        pose = self._parse_pose(raw)
        if pose is None:
            raise RuntimeError("[Robot] Could not read current pose.")
        self.robot.SetCartLinVel(APPROACH_CART_LIN_VEL)
        self.robot.SetCartAcc(APPROACH_CART_ACC)
        try:
            self.robot.SetBlending(0)
        except Exception:
            pass
        self.robot.ResumeMotion()
        self.robot.MovePose(pose["x"], pose["y"], float(z_mm),
                            TRACK_ALPHA, TRACK_BETA, TRACK_GAMMA)
        self.robot.WaitIdle()
        self._current_z = z_mm
        print(f"[Robot] At search pose Z={z_mm:.1f}.")

    def jog(self, dx: float, dy: float, dz: float):
        # Manual jog from keyboard. Safety check prevents descending below MIN_Z_MM.
        new_z = self._current_z + dz
        if new_z < MIN_Z_MM:
            print(f"[Safety] Jog blocked — Z={new_z:.1f} < MIN_Z_MM={MIN_Z_MM}")
            return
        if self.dry_run:
            print(f"[Robot] DRY_RUN jog  dX={dx:+.3f}  dY={dy:+.3f}  dZ={dz:+.3f} → Z={new_z:.1f}")
            self._current_z = new_z
            return
        self.robot.SetCartLinVel(CART_LIN_VEL)
        self.robot.SetCartAcc(CART_ACC)
        try:
            self.robot.SetBlending(0)
        except Exception:
            pass
        self.robot.ResumeMotion()
        self.robot.MoveLinRelWrf(float(dx), float(dy), float(dz), 0.0, 0.0, 0.0)
        self.robot.WaitIdle()
        self._current_z = new_z

    def servo_step(self, dx_mm: float, dy_mm: float, dz_mm: float):
        # Visual servo step — intentionally NO WaitIdle.
        # blending=100 is set before the loop starts, so moves blend into a smooth
        # trajectory. Adding WaitIdle here would stop that blending and cause jerking.
        dx_mm = float(np.clip(dx_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))
        dy_mm = float(np.clip(dy_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))
        dz_mm = float(np.clip(dz_mm, -MAX_STEP_Z_MM,  MAX_STEP_Z_MM))

        # Clamp Z so the safety floor is never breached even mid-servo
        if self._current_z + dz_mm < MIN_Z_MM:
            dz_mm = float(np.clip(MIN_Z_MM - self._current_z, -MAX_STEP_Z_MM, 0.0))

        if self.dry_run:
            print(f"[Robot] DRY_RUN servo  dX={dx_mm:+.4f}  dY={dy_mm:+.4f}  dZ={dz_mm:+.4f}")
            self._current_z += dz_mm
            return

        if SWAP_XY:
            self.robot.MoveLinRelWrf(dy_mm, dx_mm, dz_mm, 0.0, 0.0, 0.0)
        else:
            self.robot.MoveLinRelWrf(dx_mm, dy_mm, dz_mm, 0.0, 0.0, 0.0)
        self._current_z += dz_mm

    # ------------------------------------------------------------------
    # Pose averaging
    # ------------------------------------------------------------------

    def measure_stable_pose(self, n_samples=40, delay_s=0.03, settle_s=0.30):
        # Average n_samples pose readings to reduce encoder noise.
        # settle_s lets the robot stop vibrating before sampling starts.
        if self.dry_run:
            return {
                "x": 0.0, "y": 0.0, "z": self._current_z,
                "alpha": TRACK_ALPHA, "beta": TRACK_BETA, "gamma": TRACK_GAMMA,
                "std_x": 0.0, "std_y": 0.0, "std_z": 0.0,
                "std_alpha": 0.0, "std_beta": 0.0, "std_gamma": 0.0,
                "samples": n_samples,
            }
        self.robot.WaitIdle()
        time.sleep(settle_s)

        readings = []
        for _ in range(n_samples):
            try:
                raw = self.robot.GetRtCartPos()
            except Exception:
                raw = self._read_pose()
            pose = self._parse_pose(raw)
            if pose is not None:
                readings.append([pose["x"], pose["y"], pose["z"],
                                  pose["alpha"], pose["beta"], pose["gamma"]])
            time.sleep(delay_s)

        if len(readings) < 5:
            print("[Robot] Not enough valid pose samples.")
            return None

        arr = np.array(readings, dtype=float)
        avg, std = np.mean(arr, axis=0), np.std(arr, axis=0)
        result = {
            "x":         float(avg[0]), "y":         float(avg[1]), "z":         float(avg[2]),
            "alpha":     float(avg[3]), "beta":      float(avg[4]), "gamma":     float(avg[5]),
            "std_x":     float(std[0]), "std_y":     float(std[1]), "std_z":     float(std[2]),
            "std_alpha": float(std[3]), "std_beta":  float(std[4]), "std_gamma": float(std[5]),
            "samples":   len(readings),
        }
        print("[Robot] Stable pose captured:")
        print(result)
        return result

    def measure_stable_joints(self, n_samples=40, delay_s=0.03, settle_s=0.30):
        # Average joint angles over n_samples readings.
        # Joint angles are saved (not Cartesian) for MoveJoints return moves —
        # using joints avoids wrist-flip ambiguity that can occur with MoveLin/MovePose.
        if self.dry_run:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.robot.WaitIdle()
        time.sleep(settle_s)

        readings = []
        for _ in range(n_samples):
            try:
                raw = self.robot.GetRtJointPos()
            except Exception:
                raw = self._read_joints()
            joints = self._parse_joints(raw)
            if joints is not None and len(joints) == 6:
                readings.append(joints)
            time.sleep(delay_s)

        if len(readings) < 5:
            print("[Robot] Not enough valid joint samples.")
            return None

        arr = np.array(readings, dtype=float)
        avg, std = np.mean(arr, axis=0), np.std(arr, axis=0)
        result = [float(v) for v in avg]
        print("[Robot] Stable joints captured:", result)
        print("[Robot] Joint std:             ", [float(v) for v in std])
        return result

    # ------------------------------------------------------------------
    # Tool station motions
    #
    # All tool stations sit at Z=65 mm, oriented (-90, 0, 90).
    # Each tool is at a different X position along the rack:
    #   Camera  X=  0 mm
    #   OCT     X= 50 mm
    #   BioPen  X=100 mm
    #
    # Pick/place pattern for each tool:
    #   1) Move to approach position  (Y=233) — in front of the dock
    #   2) Move into the dock         (Y=260) — grab or release the tool
    #   3) Retract to approach        (Y=233) — clear of the dock
    #
    # The intermediate pose MoveJoints(90,0,0,0,0,0) swings the arm to a
    # safe posture that clears the tool rack before any large joint motion.
    # ------------------------------------------------------------------

    def pick_up_camera(self):
        if self.dry_run:
            print("[Robot] DRY_RUN pick_up_camera")
            return
        self._set_travel_speed()
        print("[Robot] Picking up camera...")
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);  self.robot.WaitIdle()
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0); self.robot.WaitIdle()
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(0, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        time.sleep(0.5)
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0); self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);  self.robot.WaitIdle()
        print("[Robot] Camera picked up.")

    def put_camera_back(self):
        if self.dry_run:
            print("[Robot] DRY_RUN put_camera_back")
            return
        self._set_travel_speed()
        print("[Robot] Returning camera...")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0); self.robot.WaitIdle()
        time.sleep(0.5)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(0, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        time.sleep(1)
        print("[Robot] Camera returned.")

    def pick_up_oct_and_go_to_target(self, target_joints: list):
        # After picking up the OCT, drive straight to the saved target position
        # using joint angles to avoid any Cartesian path singularities.
        if self.dry_run:
            print("[Robot] DRY_RUN pick_up_oct_and_go_to_target", target_joints)
            return
        self._set_travel_speed()
        print("[Robot] Picking up OCT...")
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(50, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        print("[Robot] Moving OCT to target position...")
        self.robot.MoveJoints(*target_joints); self.robot.WaitIdle()
        print("[Robot] OCT at target position.")

    def put_oct_back(self):
        if self.dry_run:
            print("[Robot] DRY_RUN put_oct_back")
            return
        self._set_travel_speed()
        print("[Robot] Returning OCT...")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(50, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        print("[Robot] OCT returned.")

    def pick_up_biopen_and_go_to_target(self, target_joints: list):
        if self.dry_run:
            print("[Robot] DRY_RUN pick_up_biopen_and_go_to_target", target_joints)
            return
        self._set_travel_speed()
        print("[Robot] Picking up BioPen...")
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # clear camera station
        time.sleep(0.5)
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);     self.robot.WaitIdle()
        print("[Robot] Moving BioPen to target position...")
        self.robot.MoveJoints(*target_joints); self.robot.WaitIdle()
        print("[Robot] BioPen at target position.")

    def put_biopen_back(self):
        if self.dry_run:
            print("[Robot] DRY_RUN put_biopen_back")
            return
        self._set_travel_speed()
        print("[Robot] Returning BioPen...")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # dock
        time.sleep(1)
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # clear
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        print("[Robot] Operation finished.")

    def close(self):
        if self.dry_run or self.robot is None:
            return
        try:
            self.robot.DeactivateRobot()
            self.robot.Disconnect()
        except Exception:
            pass


# =============================================================================
# POSE FILE HELPERS
# =============================================================================

def save_target_pose(pose: dict, joints: list, area_px: float):
    # Append this run's result to pose_reference.json.
    # Cartesian pose is saved for reference; joint angles are what we actually
    # use for MoveJoints (avoids wrist-flip issues with Cartesian targets).
    record = {
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose":       pose,
        "joints":     {f"j{i+1}": v for i, v in enumerate(joints)},
        "area_px":    round(area_px, 1),
        "note": "Joints are used for MoveJoints to avoid wrist-flip on Cartesian return.",
    }

    existing = []
    if POSE_FILE.exists():
        try:
            existing = json.loads(POSE_FILE.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except Exception:
            existing = []

    existing.append(record)
    POSE_FILE.write_text(json.dumps(existing, indent=2))
    print(f"[Pose] Saved to: {POSE_FILE}")
    return record


def print_target_summary(pose: dict, joints: list):
    print("\n" + "=" * 60)
    print("TARGET POSE CAPTURED")
    print("=" * 60)
    print("Cartesian (averaged):")
    for k in ("x", "y", "z", "alpha", "beta", "gamma"):
        print(f"  {k:5s} = {pose[k]:.6f}")
    print("----")
    print("Joint angles (used for MoveJoints return):")
    for i, v in enumerate(joints, 1):
        print(f"  j{i} = {v:.6f}")
    print("----")
    print("Std deviations (lower = more stable):")
    for k in ("std_x", "std_y", "std_z", "std_alpha", "std_beta", "std_gamma"):
        print(f"  {k:10s} = {pose[k]:.6f}")
    print(f"  samples    = {pose['samples']}")
    print("=" * 60 + "\n")


# =============================================================================
# CAMERA / FILE HELPERS
# =============================================================================

def open_camera():
    backend = cv2.CAP_DSHOW if USE_DSHOW else 0
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    return cap


def release_camera(cap, video_writer=None, video_filename=None):
    if video_writer is not None:
        video_writer.release()
        print(f"[Video] Saved: {video_filename}")
    if cap is not None:
        cap.release()
        print("[Camera] Camera OFF.")
    cv2.destroyAllWindows()


def save_picture(frame):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = SAVE_DIR / f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
    if cv2.imwrite(str(filename), frame):
        print(f"[Camera] Saved: {filename}")
    else:
        print("[Camera] Failed to save picture.")


def start_video_writer(frame_shape, fps=20.0):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = SAVE_DIR / f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    h, w     = frame_shape[:2]
    writer   = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print("[Video] Failed to start.")
        return None, None
    print(f"[Video] Recording: {filename}")
    return writer, filename


def stop_video_writer(writer, filename):
    if writer is not None:
        writer.release()
        print(f"[Video] Saved: {filename}")


# =============================================================================
# OBJECT DETECTION
# =============================================================================

def detect_object(frame_bgr):
    # Detect by grayscale intensity range [I_MIN, I_MAX].
    # Returns (contour, centroid, area_px, bbox, mask).
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(blur, I_MIN, I_MAX)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area_px     = float(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or area_px < MIN_CONTOUR_AREA:
        return None, None, None, None, mask

    best = max(contours, key=cv2.contourArea)
    M    = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, None, None, mask

    centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=float)
    bbox     = cv2.boundingRect(best)
    return best, centroid, area_px, bbox, mask


# =============================================================================
# HUD OVERLAY
# =============================================================================

def draw_hud(display, state, centroid_f, err_norm, area_px, area_target, robot_z, target_pose):
    color = _STATE_COLORS.get(state, (255, 255, 255))
    lines = [
        (f"State: {state.value}", color),
        (f"Centroid: ({centroid_f[0]:.1f}, {centroid_f[1]:.1f})  err={err_norm:.1f} px",
         (0, 255, 255)),
        (f"Area: {area_px:.0f}  target: {area_target:.0f}", (0, 255, 255)),
        (f"Z approx: {robot_z:.1f} mm", (200, 200, 200)),
    ]
    if target_pose is not None:
        lines += [
            (f"Target x={target_pose['x']:.2f}  y={target_pose['y']:.2f}  z={target_pose['z']:.2f}",
             (180, 255, 180)),
            (f"       a={target_pose['alpha']:.2f}  b={target_pose['beta']:.2f}  g={target_pose['gamma']:.2f}",
             (180, 255, 180)),
        ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (10, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)


# =============================================================================
# HOLD SEQUENCE
# Triggered once when HOLD state is first reached.
# Captures the stable target pose, then executes the full tool swap:
#   camera back → OCT to target → OCT back → BioPen to target → BioPen back
# The camera MUST be released before tool motions start (robot needs to move freely).
# =============================================================================

def run_tool_sequence(robot, area_px, cap, video_writer, video_filename):
    print("[Pose] HOLD reached — capturing stable target pose...")

    target_pose   = robot.measure_stable_pose(POSE_AVG_SAMPLES, POSE_AVG_DELAY_S, POSE_SETTLE_S)
    target_joints = robot.measure_stable_joints(POSE_AVG_SAMPLES, POSE_AVG_DELAY_S, POSE_SETTLE_S)

    if target_pose is None or target_joints is None:
        return None, None

    save_target_pose(target_pose, target_joints, area_px)
    time.sleep(DELAY_AFTER_POSE_CAPTURE_S)

    print_target_summary(target_pose, target_joints)
    time.sleep(DELAY_AFTER_PRINT_SUMMARY_S)

    # Release camera before robot moves — camera cable must not restrict motion
    print("[Main] Releasing camera before tool swap...")
    release_camera(cap, video_writer, video_filename)

    # Tool sequence: camera back → OCT → BioPen
    robot.put_camera_back()

    robot.pick_up_oct_and_go_to_target(target_joints)
    time.sleep(DELAY_AFTER_OCT_AT_TARGET_S)
    robot.put_oct_back()

    robot.pick_up_biopen_and_go_to_target(target_joints)
    time.sleep(DELAY_AFTER_BIOPEN_AT_TARGET_S)
    robot.put_biopen_back()

    return target_pose, target_joints


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # Full sequence:
    #   1) Pick up camera
    #   2) Move to search pose and start tracking loop
    #   3) WAIT_DETECT → ALIGN_XY → APPROACH_Z → HOLD
    #   4) At HOLD: capture stable pose, release camera
    #   5) Camera back → OCT to target → OCT back → BioPen to target → BioPen back
    # ------------------------------------------------------------------
    robot = Meca500Client(ROBOT_IP, dry_run=DRY_RUN)

    cap            = None
    video_writer   = None
    video_filename = None
    is_recording   = False

    centroid_f    = None
    step_xy_f     = np.zeros(2, dtype=float)
    step_z_f      = 0.0
    next_time     = time.time()

    state           = State.WAIT_DETECT
    align_frame_count = 0
    area_ref        = 0.0
    area_target     = AREA_TARGET

    target_pose     = None
    sequence_done   = False
    camera_released = False

    try:
        # ---- Hardware init ----
        robot.connect()
        robot.pick_up_camera()
        robot.move_zero_joints()
        print(f"[Main] Waiting {DELAY_AFTER_ZERO_S:.1f} s...")
        time.sleep(DELAY_AFTER_ZERO_S)

        robot.move_to_search_pose(SEARCH_Z)
        robot._set_tracking_speed()

        cap = open_camera()
        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Could not read from camera.")

        h0, w0    = frame0.shape[:2]
        target_px = np.array([w0 / 2.0, h0 / 2.0], dtype=float)

        print("\n=== Tracking started ===")
        print("Flow: WAIT_DETECT → ALIGN_XY → APPROACH_Z → HOLD → tool sequence")
        print(f"AREA_TARGET={area_target:.0f}  AREA_TOLERANCE={AREA_TOLERANCE:.0f}")
        print("Keys: z=+Z  s=-Z  t=set area target  p=picture  v=video  q=quit\n")

        while True:
            # ---- Fixed-rate timing ----
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += DT

            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame read failed.")
                break

            display = frame.copy()
            cv2.circle(display,
                       (int(target_px[0]), int(target_px[1])),
                       CAM_CENTER_RADIUS_PX, (0, 0, 0), CAM_CENTER_THICKNESS)

            contour, centroid, area_px, bbox, mask = detect_object(frame)

            # ==================================================================
            # OBJECT NOT DETECTED
            # ==================================================================
            if contour is None:
                centroid_f        = None
                step_xy_f[:]      = 0.0
                step_z_f          = 0.0
                align_frame_count = 0
                state             = State.WAIT_DETECT
                cv2.putText(display, "WAIT_DETECT — no object",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ==================================================================
            # OBJECT DETECTED
            # ==================================================================
            else:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                x_b, y_b, w_b, h_b = bbox
                cv2.rectangle(display, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 255, 0), 1)

                # Smooth centroid to reduce pixel noise
                if centroid_f is None:
                    centroid_f = centroid.copy()
                else:
                    centroid_f = (1.0 - ALPHA_CENTROID) * centroid_f + ALPHA_CENTROID * centroid

                cv2.drawMarker(display,
                               (int(centroid_f[0]), int(centroid_f[1])),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                err_px   = centroid_f - target_px
                err_norm = float(np.linalg.norm(err_px))

                # Wider deadband during approach to avoid fighting the Z motion
                deadband     = DEADBAND_PX_APPROACH if state == State.APPROACH_Z else DEADBAND_PX
                ctrl_err     = err_px.copy()
                if err_norm < deadband:
                    ctrl_err[:] = 0.0

                # Gain scaling: as the robot descends the object looks bigger, so
                # the same pixel error represents a smaller real-world distance.
                # Scale gain down proportionally so corrections don't overshoot.
                gain_scale = 1.0
                if state == State.APPROACH_Z and area_ref > 0.0:
                    gain_scale = float(np.clip(
                        np.sqrt(area_ref / max(area_px, 1.0)), 0.25, 2.0))

                step_xy   = -(GAIN_XY * gain_scale) * pixel_to_mm.dot(ctrl_err)
                if FLIP_X: step_xy[0] *= -1.0
                if FLIP_Y: step_xy[1] *= -1.0
                step_xy_f = (1.0 - ALPHA_STEP_XY) * step_xy_f + ALPHA_STEP_XY * step_xy
                xy_aligned = err_norm <= CENTER_ACCEPT_PX

                # ---- State machine ----
                if state == State.WAIT_DETECT:
                    state = State.ALIGN_XY

                elif state == State.ALIGN_XY:
                    if xy_aligned:
                        align_frame_count += 1
                        if align_frame_count >= XY_ALIGN_HOLD_FRAMES:
                            area_ref  = area_px
                            state     = State.APPROACH_Z
                            print(f"[State] → APPROACH_Z  area_ref={area_ref:.0f}")
                    else:
                        align_frame_count = 0

                elif state == State.APPROACH_Z:
                    if not xy_aligned:
                        # Lost alignment — back up and re-center before descending again
                        state             = State.ALIGN_XY
                        align_frame_count = 0
                        step_z_f          = 0.0
                    else:
                        if abs(area_target - area_px) <= AREA_TOLERANCE:
                            step_z_f     = 0.0
                            step_xy_f[:] = 0.0
                            state        = State.HOLD
                            print("[State] → HOLD")

                elif state == State.HOLD:
                    step_xy_f[:] = 0.0
                    step_z_f     = 0.0

                # ---- Compute control outputs ----
                step_xy_cmd = step_xy_f.copy() if AUTO_XY_ON else np.zeros(2)
                step_z_cmd  = 0.0

                if state == State.APPROACH_Z and AUTO_Z_ON:
                    area_error  = area_target - area_px
                    raw_z       = -GAIN_Z_AREA * area_error
                    if FLIP_Z: raw_z *= -1.0
                    raw_z       = float(np.clip(raw_z, -MAX_STEP_Z_MM, MAX_STEP_Z_MM))
                    step_z_f    = (1.0 - ALPHA_STEP_Z) * step_z_f + ALPHA_STEP_Z * raw_z
                    step_z_cmd  = float(step_z_f)

                if np.linalg.norm(step_xy_cmd) > 1e-9 or abs(step_z_cmd) > 1e-9:
                    robot.servo_step(float(step_xy_cmd[0]), float(step_xy_cmd[1]), step_z_cmd)

                # ---- HOLD: run tool sequence once, then exit ----
                if state == State.HOLD and not sequence_done:
                    target_pose, _ = run_tool_sequence(
                        robot, area_px, cap, video_writer, video_filename
                    )
                    if target_pose is not None:
                        cap = video_writer = video_filename = None
                        is_recording    = False
                        camera_released = True
                        sequence_done   = True
                        print("[Main] All tools done. Exiting.")
                        break

                draw_hud(display, state, centroid_f, err_norm,
                         area_px, area_target, robot._current_z, target_pose)

            if is_recording and video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Camera", display)
            cv2.imshow("Mask",   mask)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("z"): robot.jog(0.0, 0.0, +JOG_STEP_MM)
            elif key == ord("s"): robot.jog(0.0, 0.0, -JOG_STEP_MM)
            elif key == ord("t"):
                if contour is not None:
                    area_target = area_px
                    print(f"[Target] Area target set to {area_target:.0f} px²")
            elif key == ord("p"): save_picture(display)
            elif key == ord("v"):
                if not is_recording:
                    video_writer, video_filename = start_video_writer(
                        display.shape, fps=max(1.0, 1.0 / DT))
                    if video_writer is not None:
                        is_recording = True
                else:
                    stop_video_writer(video_writer, video_filename)
                    video_writer = video_filename = None
                    is_recording = False

    finally:
        # Camera may already be released by run_tool_sequence — only release if not done yet
        if not camera_released:
            if video_writer is not None:
                stop_video_writer(video_writer, video_filename)
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
