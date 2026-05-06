import cv2
import time
import json
import numpy as np
from enum import Enum
from pathlib import Path

# CONFIG-----------------------------------------------
ROBOT_IP  = "192.168.0.100"
CAM_INDEX = 0
USE_DSHOW = True     # more stable on Windows
DRY_RUN   = False    # set True to test without moving the robot

ZERO_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

JOINT_VEL_LIMIT       = 25
CART_LIN_VEL          = 400
CART_ACC              = 250

TRACK_CART_LIN_VEL    = 200  # speed once tracking starts
TRACK_CART_ACC        = 400

APPROACH_CART_LIN_VEL = 150  # slower speed when moving to search pose
APPROACH_CART_ACC     = 120

DELAY_AFTER_ZERO_S             = 1.0
DELAY_AFTER_POSE_CAPTURE_S     = 2.0
DELAY_AFTER_PRINT_SUMMARY_S    = 2.0
DELAY_AFTER_OCT_AT_TARGET_S    = 1.0
DELAY_AFTER_BIOPEN_AT_TARGET_S = 1.0

JOG_STEP_MM = 1.0   # mm per keyboard press
MIN_Z_MM    = 100.0  # robot will never go below this Z

SAVE_DIR  = Path(__file__).parent / "pic"
POSE_FILE = Path(__file__).parent / "pose_reference.json"

# Object DETECTION----------------------------------------------------------
I_MIN            = 30
I_MAX            = 100
MIN_CONTOUR_AREA = 200   # ignore blobs smaller than this

CAM_CENTER_RADIUS_PX = 10
CAM_CENTER_THICKNESS = 2

# TRACKING POSE (where the robot waits and starts looking for the object)
SEARCH_Z    = 250.0
TRACK_ALPHA =   0.0
TRACK_BETA  =  90.0   # tool pointing straight down
TRACK_GAMMA =   0.0

# XY CONTROL------------------------
DT         = 0.05
AUTO_XY_ON = True

DEADBAND_PX          =  6.0
DEADBAND_PX_APPROACH = 10.0   # bigger deadband during Z descent to avoid fighting it
CENTER_ACCEPT_PX     =  8.0   # within this = "aligned"

ALPHA_CENTROID = 0.35
ALPHA_STEP_XY  = 0.46
GAIN_XY        = 0.30

# calibrated with pixel_calibration.py
MM_PER_PX_X = 0.03
MM_PER_PX_Y = 0.03

pixel_to_mm = np.array([
    [MM_PER_PX_X, 0.0        ],
    [0.0,         MM_PER_PX_Y],
], dtype=float)

MAX_STEP_XY_MM       = 0.5
FLIP_X               = False
FLIP_Y               = False
SWAP_XY              = True   # camera is rotated so X and Y are swapped
XY_ALIGN_HOLD_FRAMES = 5      # frames XY must stay aligned before Z descent begins

# Z CONTROL BY AREA--------------------------------------------------------------------
AUTO_Z_ON      = True   # can also jog manually with S / Z keys
AREA_TARGET    = 70000.0
AREA_TOLERANCE =  1000.0
FLIP_Z         = False

GAIN_Z_AREA   = 0.00012
ALPHA_STEP_Z  = 0.55
MAX_STEP_Z_MM = 0.8

# FINAL POSE AVERAGING
# once HOLD is reached, we average several samples to get a stable pose
# before driving the tools there
POSE_AVG_SAMPLES = 40
POSE_AVG_DELAY_S = 0.03
POSE_SETTLE_S    = 0.30   # wait for vibrations to die down before sampling

# STATES--------------------------------------------------------------------
class State(Enum):
    WAIT_DETECT = "WAIT_DETECT"
    ALIGN_XY    = "ALIGN_XY"
    APPROACH_Z  = "APPROACH_Z"
    HOLD        = "HOLD"

_STATE_COLORS = {
    State.ALIGN_XY:   (0, 255, 255),
    State.APPROACH_Z: (0, 165, 255),
    State.HOLD:       (0, 255, 0),
}


# ROBOT WRAPPER-----------------------------------------------------------
class Meca500Client:
    def __init__(self, ip: str, dry_run: bool = True):
        self.ip         = ip
        self.dry_run    = dry_run
        self.robot      = None
        self._current_z = SEARCH_Z  # track Z locally to avoid reading pose every servo step

    @staticmethod
    def _parse_pose(raw):
        # API returns 6 or 7 values depending on firmware — always take the last 6
        if raw is None:
            return None
        raw = list(raw)
        if len(raw) >= 7:
            values = raw[-6:]
        elif len(raw) >= 6:
            values = raw[:6]
        else:
            return None
        return {
            "x":     float(values[0]),
            "y":     float(values[1]),
            "z":     float(values[2]),
            "alpha": float(values[3]),
            "beta":  float(values[4]),
            "gamma": float(values[5]),
        }

    @staticmethod
    def _parse_joints(raw):
        if raw is None:
            return None
        raw = list(raw)
        if len(raw) >= 7:
            values = raw[-6:]
        elif len(raw) >= 6:
            values = raw[:6]
        else:
            return None
        return [float(v) for v in values]

    def _read_pose(self):
        # method name changed across firmware versions, so try all known ones
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

    def connect(self):
        if self.dry_run:
            print("[Robot] DRY_RUN=True -> not connecting.")
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
            self.robot.WaitIdle()   # older firmware doesn't have WaitHomed
        self.robot.ResumeMotion()
        print("[Robot] Connected + homed.")

    def _set_travel_speed(self):
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

    def pick_up_camera(self):
        """Pick up the camera before vision-guided tracking."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Taking the camera")
            return
        self._set_travel_speed()
        print("[Robot] # Taking the camera")
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);  self.robot.WaitIdle()
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(0, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        time.sleep(0.5)
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);  self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        print("[Robot] Camera taken.")

    def put_camera_back(self):
        """Return camera to its station after target pose is captured."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Put the camera back")
            return
        self._set_travel_speed()
        print("[Robot] # Put the camera back")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);  self.robot.WaitIdle()
        time.sleep(0.5)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(0, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        time.sleep(1)
        print("[Robot] Camera returned.")

    def pick_up_oct_and_go_to_target(self, target_joints: list):
        """Pick up OCT probe and drive to the saved target position."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Taking the OCT")
            print("[Robot] DRY_RUN MoveJoints to target_joints:", target_joints)
            return
        self._set_travel_speed()
        print("[Robot] # Taking the OCT")
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(50, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        print("[Robot] Moving OCT to target...")
        self.robot.MoveJoints(*target_joints); self.robot.WaitIdle()
        print("[Robot] OCT at target.")

    def put_oct_back(self):
        """Return OCT probe to its station."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Put OCT back")
            return
        self._set_travel_speed()
        print("[Robot] # Put OCT back")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);   self.robot.WaitIdle()
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(50, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(50, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        print("[Robot] OCT returned.")

    def pick_up_biopen_and_go_to_target(self, target_joints: list):
        """Pick up BioPen and drive to the saved target position."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Take BioPen")
            print("[Robot] DRY_RUN MoveJoints to target_joints:", target_joints)
            return
        self._set_travel_speed()
        print("[Robot] # Take BioPen")
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # clear camera station
        time.sleep(0.5)
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        self.robot.MoveJoints(0, 0, 0, 0, 0, 0);     self.robot.WaitIdle()
        print("[Robot] Moving BioPen to target...")
        self.robot.MoveJoints(*target_joints); self.robot.WaitIdle()
        print("[Robot] BioPen at target.")

    def put_biopen_back(self):
        """Return BioPen to its station."""
        if self.dry_run:
            print("[Robot] DRY_RUN # Put BioPen back")
            return
        self._set_travel_speed()
        print("[Robot] # Put BioPen back")
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveLin(100, 260, 65, -90, 0, 90); self.robot.WaitIdle()  # into dock
        time.sleep(1)
        self.robot.MoveLin(100, 233, 65, -90, 0, 90); self.robot.WaitIdle()  # retract
        self.robot.MoveLin(  0, 233, 65, -90, 0, 90); self.robot.WaitIdle()
        self.robot.MoveJoints(90, 0, 0, 0, 0, 0);    self.robot.WaitIdle()
        print("[Robot] Operation Finished.")

    def move_zero_joints(self):
        if self.dry_run:
            print(f"[Robot] DRY_RUN MoveJoints{ZERO_JOINTS}")
            return
        self._set_travel_speed()
        self.robot.MoveJoints(*ZERO_JOINTS)
        self.robot.WaitIdle()
        print("[Robot] Reached zero joints.")

    def _set_tracking_speed(self):
        if self.dry_run or self.robot is None:
            return
        self.robot.SetCartLinVel(TRACK_CART_LIN_VEL)
        self.robot.SetCartAcc(TRACK_CART_ACC)
        try:
            self.robot.SetBlending(100)  # blending=100 keeps motion smooth between servo steps
        except Exception:
            pass
        self.robot.ResumeMotion()
        print("[Robot] Tracking params set (vel/acc/blending=100).")

    def move_to_search_pose(self, z_mm: float):
        if self.dry_run:
            print(f"[Robot] DRY_RUN MovePose Z={z_mm:.3f}")
            self._current_z = z_mm
            return
        raw  = self._read_pose()
        pose = self._parse_pose(raw)
        if pose is None:
            raise RuntimeError("[Robot] Could not read current pose.")
        x, y = pose["x"], pose["y"]
        self.robot.SetCartLinVel(APPROACH_CART_LIN_VEL)
        self.robot.SetCartAcc(APPROACH_CART_ACC)
        try:
            self.robot.SetBlending(0)
        except Exception:
            pass
        self.robot.ResumeMotion()
        self.robot.MovePose(x, y, float(z_mm), TRACK_ALPHA, TRACK_BETA, TRACK_GAMMA)
        self.robot.WaitIdle()
        self._current_z = z_mm
        print(f"[Robot] At search pose Z={z_mm:.3f}.")

    def jog(self, dx: float, dy: float, dz: float):
        new_z = self._current_z + dz
        if new_z < MIN_Z_MM:
            print(f"[Safety] Jog blocked — Z={new_z:.1f} < MIN_Z_MM={MIN_Z_MM}")
            return
        if self.dry_run:
            print(f"[Robot] DRY_RUN Jog dZ={dz:+.3f} -> Z={new_z:.3f}")
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
        # no WaitIdle here on purpose — blending=100 keeps moves continuous
        dx_mm = float(np.clip(dx_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))
        dy_mm = float(np.clip(dy_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))
        dz_mm = float(np.clip(dz_mm, -MAX_STEP_Z_MM,  MAX_STEP_Z_MM))

        if self._current_z + dz_mm < MIN_Z_MM:
            dz_mm = float(np.clip(MIN_Z_MM - self._current_z, -MAX_STEP_Z_MM, 0.0))

        if self.dry_run:
            print(f"[Robot] DRY_RUN servo_step dX={dx_mm:+.4f} dY={dy_mm:+.4f} dZ={dz_mm:+.4f}")
            self._current_z += dz_mm
            return

        if SWAP_XY:
            self.robot.MoveLinRelWrf(dy_mm, dx_mm, dz_mm, 0.0, 0.0, 0.0)
        else:
            self.robot.MoveLinRelWrf(dx_mm, dy_mm, dz_mm, 0.0, 0.0, 0.0)
        self._current_z += dz_mm

    def measure_stable_pose(self, n_samples=40, delay_s=0.03, settle_s=0.30):
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
                readings.append([
                    pose["x"], pose["y"], pose["z"],
                    pose["alpha"], pose["beta"], pose["gamma"],
                ])
            time.sleep(delay_s)

        if len(readings) < 5:
            print("[Robot] Not enough valid pose samples.")
            return None

        arr = np.array(readings, dtype=float)
        avg = np.mean(arr, axis=0)
        std = np.std(arr,  axis=0)
        result = {
            "x":         float(avg[0]), "y":         float(avg[1]), "z":         float(avg[2]),
            "alpha":     float(avg[3]), "beta":      float(avg[4]), "gamma":     float(avg[5]),
            "std_x":     float(std[0]), "std_y":     float(std[1]), "std_z":     float(std[2]),
            "std_alpha": float(std[3]), "std_beta":  float(std[4]), "std_gamma": float(std[5]),
            "samples":   len(readings),
        }
        print("[Robot] Stable averaged target pose:")
        print(result)
        return result

    def measure_stable_joints(self, n_samples=40, delay_s=0.03, settle_s=0.30):
        # we save joint angles (not Cartesian) for the return move
        # because MoveJoints avoids wrist-flip issues that can happen with MoveLin
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
        avg = np.mean(arr, axis=0)
        std = np.std(arr,  axis=0)
        result = [float(v) for v in avg]
        print("[Robot] Stable averaged target joints:")
        print(result)
        print("[Robot] Joint std:", [float(v) for v in std])
        return result

    def close(self):
        if self.dry_run or self.robot is None:
            return
        try:
            self.robot.DeactivateRobot()
            self.robot.Disconnect()
        except Exception:
            pass


# SAVE FINAL REFERENCE POSE (target joints for future movements)---------------
def save_target_pose(pose: dict, joints: list, area_px: float):
    record = {
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "pose_type":  "target_pose_stable_average_GetRtCartPos",
        "joint_type": "target_joints_stable_average_GetRtJointPos",
        "pose":       pose,
        "joints": {
            "j1": joints[0], "j2": joints[1], "j3": joints[2],
            "j4": joints[3], "j5": joints[4], "j6": joints[5],
        },
        "area_px": round(area_px, 1),
        "note": (
            "Cartesian pose saved for reference. "
            "Joints used for MoveJoints return to avoid wrist flipping."
        ),
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
    print("\n============================================================")
    print("TARGET POSE + JOINTS CAPTURED AS REFERENCE")
    print("============================================================")
    print("Cartesian:")
    print(f"  x     = {pose['x']:.6f}")
    print(f"  y     = {pose['y']:.6f}")
    print(f"  z     = {pose['z']:.6f}")
    print(f"  alpha = {pose['alpha']:.6f}")
    print(f"  beta  = {pose['beta']:.6f}")
    print(f"  gamma = {pose['gamma']:.6f}")
    print("------------------------------------------------------------")
    print("Joints:")
    for i, v in enumerate(joints, 1):
        print(f"  j{i} = {v:.6f}")
    print("------------------------------------------------------------")
    print("Cartesian std (how stable the pose was):")
    print(f"  std_x     = {pose['std_x']:.6f}")
    print(f"  std_y     = {pose['std_y']:.6f}")
    print(f"  std_z     = {pose['std_z']:.6f}")
    print(f"  std_alpha = {pose['std_alpha']:.6f}")
    print(f"  std_beta  = {pose['std_beta']:.6f}")
    print(f"  std_gamma = {pose['std_gamma']:.6f}")
    print(f"  samples   = {pose['samples']}")
    print("============================================================\n")


# CAMERA HELPERS------------------------------------------------------------------------------
def open_camera():
    backend = cv2.CAP_DSHOW if USE_DSHOW else 0
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    return cap


def release_camera(cap, video_writer=None, video_filename=None):
    if video_writer is not None:
        video_writer.release()
        print(f"[Video] Stopped: {video_filename}")
    if cap is not None:
        cap.release()
        print("[Camera] Camera OFF.")
    cv2.destroyAllWindows()


def save_picture(frame):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fname = SAVE_DIR / f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
    ok    = cv2.imwrite(str(fname), frame)
    print(f"[Camera] {'Saved: ' + str(fname) if ok else 'Failed to save image.'}")


def start_video_writer(frame_shape, fps=20.0):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fname  = SAVE_DIR / f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    h, w   = frame_shape[:2]
    writer = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print("[Video] Failed to start video writer.")
        return None, None
    print(f"[Video] Recording: {fname}")
    return writer, fname


def stop_video_writer(writer, filename):
    if writer is not None:
        writer.release()
        print(f"[Video] Stopped: {filename}")


# OBJECT DETECTION---------------------------------------------------
def detect_object(frame_bgr):
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


# HUD OVERLAY---------------------------------------------------------------------------
def draw_hud(display, state, centroid_f, err_norm, area_px, area_target, robot_z, target_pose):
    color = _STATE_COLORS.get(state, (255, 255, 255))
    cv2.putText(display, f"State: {state.value}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display,
                f"Center: ({centroid_f[0]:.1f}, {centroid_f[1]:.1f})  err={err_norm:.1f}px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(display,
                f"Area: {area_px:.0f}  target: {area_target:.0f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(display,
                f"Z approx: {robot_z:.1f} mm",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 2)
    if target_pose is not None:
        cv2.putText(display,
                    f"Target x={target_pose['x']:.2f} y={target_pose['y']:.2f} z={target_pose['z']:.2f}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2)
        cv2.putText(display,
                    f"       a={target_pose['alpha']:.2f} b={target_pose['beta']:.2f} g={target_pose['gamma']:.2f}",
                    (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2)


# HOLD SEQUENCE-----------------------------------------------------------------------
def run_tool_sequence(robot, area_px, cap, video_writer, video_filename):
    """Capture target pose then run the full tool swap: OCT then BioPen."""
    print("[Pose] HOLD reached. Capturing stable averaged target pose and joints...")

    target_pose   = robot.measure_stable_pose(POSE_AVG_SAMPLES, POSE_AVG_DELAY_S, POSE_SETTLE_S)
    target_joints = robot.measure_stable_joints(POSE_AVG_SAMPLES, POSE_AVG_DELAY_S, POSE_SETTLE_S)

    if target_pose is None or target_joints is None:
        return None, None

    save_target_pose(target_pose, target_joints, area_px)
    time.sleep(DELAY_AFTER_POSE_CAPTURE_S)

    print_target_summary(target_pose, target_joints)
    time.sleep(DELAY_AFTER_PRINT_SUMMARY_S)

    # camera must be released before robot starts moving to other stations
    print("[Main] Turning camera OFF before tool motions...")
    release_camera(cap, video_writer, video_filename)

    robot.put_camera_back()

    robot.pick_up_oct_and_go_to_target(target_joints)
    time.sleep(DELAY_AFTER_OCT_AT_TARGET_S)
    robot.put_oct_back()

    robot.pick_up_biopen_and_go_to_target(target_joints)
    time.sleep(DELAY_AFTER_BIOPEN_AT_TARGET_S)
    robot.put_biopen_back()

    return target_pose, target_joints


# MAIN-----------------------------------------------------------
def main():
    robot = Meca500Client(ROBOT_IP, dry_run=DRY_RUN)

    cap            = None
    video_writer   = None
    video_filename = None
    is_recording   = False

    centroid_f = None
    step_xy_f  = np.zeros(2, dtype=float)
    step_z_f   = 0.0
    next_time  = time.time()

    state             = State.WAIT_DETECT
    align_frame_count = 0
    area_ref          = 0.0
    area_target       = AREA_TARGET

    target_pose     = None
    sequence_done   = False
    camera_released = False

    try:
        robot.connect()
        robot.pick_up_camera()
        robot.move_zero_joints()

        print(f"[Main] Delay: {DELAY_AFTER_ZERO_S:.2f} s")
        time.sleep(DELAY_AFTER_ZERO_S)

        robot.move_to_search_pose(SEARCH_Z)
        robot._set_tracking_speed()

        cap = open_camera()

        print("[Main] Ready.")
        print("[Main] States: WAIT_DETECT → ALIGN_XY → APPROACH_Z → HOLD")
        print(f"[Main] AREA_TARGET = {area_target:.1f}  AREA_TOLERANCE = {AREA_TOLERANCE:.1f}")
        print(f"[Main] Pose reference file: {POSE_FILE}")

        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Could not read from camera.")

        h0, w0    = frame0.shape[:2]
        target_px = np.array([w0 / 2.0, h0 / 2.0], dtype=float)

        while True:
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += DT

            ret, frame = cap.read()
            if not ret:
                print("[Camera] Failed to read frame.")
                break

            display        = frame.copy()
            cam_cx, cam_cy = int(target_px[0]), int(target_px[1])
            cv2.circle(display, (cam_cx, cam_cy), CAM_CENTER_RADIUS_PX, (0, 0, 0), CAM_CENTER_THICKNESS)

            contour, centroid, area_px, bbox, mask = detect_object(frame)

            if contour is None:
                centroid_f        = None
                step_xy_f[:]      = 0.0
                step_z_f          = 0.0
                align_frame_count = 0
                state             = State.WAIT_DETECT
                cv2.putText(display, "WAIT_DETECT - no object", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                x, y, w_box, h_box = bbox
                cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (255, 255, 0), 1)

                if centroid_f is None:
                    centroid_f = centroid.copy()
                else:
                    centroid_f = (1.0 - ALPHA_CENTROID) * centroid_f + ALPHA_CENTROID * centroid

                cv2.drawMarker(display, (int(centroid_f[0]), int(centroid_f[1])),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                err_px   = centroid_f - target_px
                err_norm = float(np.linalg.norm(err_px))

                deadband     = DEADBAND_PX_APPROACH if state == State.APPROACH_Z else DEADBAND_PX
                err_for_ctrl = err_px.copy()
                if err_norm < deadband:
                    err_for_ctrl[:] = 0.0

                # scale gain down as robot gets closer — bigger area = same pixel error is a smaller real distance
                if state == State.APPROACH_Z and area_ref > 0.0:
                    gain_scale = float(np.clip(
                        np.sqrt(area_ref / max(area_px, 1.0)), 0.25, 2.0))
                else:
                    gain_scale = 1.0

                step_xy = -(GAIN_XY * gain_scale) * pixel_to_mm.dot(err_for_ctrl)
                if FLIP_X: step_xy[0] *= -1.0
                if FLIP_Y: step_xy[1] *= -1.0

                step_xy_f  = (1.0 - ALPHA_STEP_XY) * step_xy_f + ALPHA_STEP_XY * step_xy
                xy_aligned = err_norm <= CENTER_ACCEPT_PX

                # state machine -------------------------------------------------------
                if state == State.WAIT_DETECT:
                    state = State.ALIGN_XY

                elif state == State.ALIGN_XY:
                    if xy_aligned:
                        align_frame_count += 1
                        if align_frame_count >= XY_ALIGN_HOLD_FRAMES:
                            area_ref  = area_px
                            state     = State.APPROACH_Z
                            print(f"[State] -> APPROACH_Z  area_ref={area_ref:.1f}")
                    else:
                        align_frame_count = 0

                elif state == State.APPROACH_Z:
                    if not xy_aligned:
                        state             = State.ALIGN_XY
                        align_frame_count = 0
                        step_z_f          = 0.0
                    else:
                        area_error = area_target - area_px
                        if abs(area_error) <= AREA_TOLERANCE:
                            step_z_f     = 0.0
                            step_xy_f[:] = 0.0
                            state        = State.HOLD
                            print("[State] -> HOLD")

                elif state == State.HOLD:
                    step_xy_f[:] = 0.0
                    step_z_f     = 0.0

                # control outputs---------------------------------------------------------------
                step_xy_cmd = step_xy_f.copy() if AUTO_XY_ON else np.zeros(2)
                step_z_cmd  = 0.0

                if state == State.APPROACH_Z and AUTO_Z_ON:
                    area_error  = area_target - area_px
                    raw_step_z  = -GAIN_Z_AREA * area_error
                    if FLIP_Z: raw_step_z *= -1.0
                    raw_step_z  = float(np.clip(raw_step_z, -MAX_STEP_Z_MM, MAX_STEP_Z_MM))
                    step_z_f    = (1.0 - ALPHA_STEP_Z) * step_z_f + ALPHA_STEP_Z * raw_step_z
                    step_z_cmd  = float(step_z_f)

                if np.linalg.norm(step_xy_cmd) > 1e-9 or abs(step_z_cmd) > 1e-9:
                    robot.servo_step(float(step_xy_cmd[0]), float(step_xy_cmd[1]), step_z_cmd)

                # HOLD: capture pose and run tool sequence-------------------------------------------------
                if state == State.HOLD and not sequence_done:
                    target_pose, _ = run_tool_sequence(
                        robot, area_px, cap, video_writer, video_filename
                    )
                    if target_pose is not None:
                        cap = video_writer = video_filename = None
                        is_recording    = False
                        camera_released = True
                        sequence_done   = True
                        print("[Main] Operation Finished.")
                        break

                draw_hud(display, state, centroid_f, err_norm,
                         area_px, area_target, robot._current_z, target_pose)

            if is_recording and video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Camera", display)
            cv2.imshow("Mask",   mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                robot.jog(0.0, 0.0, +JOG_STEP_MM)
            elif key == ord("s"):
                robot.jog(0.0, 0.0, -JOG_STEP_MM)
            elif key == ord("t"):
                if contour is not None:
                    area_target = area_px
                    print(f"[Target] Area target updated to {area_target:.1f} px²")
            elif key == ord("p"):
                save_picture(display)
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
        if not camera_released:
            if video_writer is not None:
                stop_video_writer(video_writer, video_filename)
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
