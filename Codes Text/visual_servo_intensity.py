import cv2
import time
import numpy as np
from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================

ROBOT_IP  = "192.168.0.100"
CAM_INDEX = 0        # camera index (0 = first USB camera)
USE_DSHOW = True     # DirectShow backend — usually more stable on Windows
DRY_RUN   = False    # True = print robot commands without moving the robot (safe testing mode)

ZERO_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# Travel speeds — used when homing and moving to the search pose
JOINT_VEL_LIMIT       = 25
TRAVEL_CART_LIN_VEL   = 400
TRAVEL_CART_ACC       = 250
APPROACH_CART_LIN_VEL = 150
APPROACH_CART_ACC     = 120

# Tracking speeds — used during the servo control loop (XY + Z moves)
# High TRACK_CART_ACC makes each small step complete within DT so the loop
# stays in sync with the robot's actual position and moves don't pile up.
TRACK_CART_LIN_VEL = 120   # mm/s
TRACK_CART_ACC     = 400   # mm/s²
TRACK_BLENDING     = 0     # 0 = sharp stop at each step (no queuing of stale moves)

DELAY_AFTER_ZERO_S = 1.0   # pause after homing before moving to search pose
JOG_STEP_MM        = 1.0   # how far each manual Z keypress moves the robot (mm)

# Pictures and video are saved here (folder is created automatically if missing)
SAVE_DIR = Path(__file__).resolve().parent / "pictures"


# =============================================================================
# INTENSITY-BASED DETECTION
# =============================================================================

# Object is detected by finding pixels whose grayscale brightness falls in [I_MIN, I_MAX].
# Tune these to isolate your target from the background.
I_MIN            = 30
I_MAX            = 100
MIN_CONTOUR_AREA = 200   # blobs smaller than this (px²) are ignored as noise

# Small crosshair drawn at the image center so you can see where the robot is aiming
CAM_CENTER_RADIUS_PX = 10
CAM_CENTER_THICKNESS = 2


# =============================================================================
# SEARCH POSE
# =============================================================================

# The robot moves here first before the control loop starts.
# SEARCH_Z is the height (mm) from which the robot begins looking for the object.
SEARCH_Z    = 250.0
TRACK_ALPHA =   0.0
TRACK_BETA  =  90.0
TRACK_GAMMA =   0.0

MIN_Z_MM = 100.0   # hard safety floor — robot will never descend below this Z


# =============================================================================
# XY CONTROL
# =============================================================================

DT = 0.10   # control loop period in seconds (10 Hz)

# Deadband: pixel errors smaller than this are treated as zero (prevents jitter).
# A larger deadband is used during Z approach to avoid over-correcting when close.
DEADBAND_PX          =  6.0
DEADBAND_PX_APPROACH = 10.0

CENTER_ACCEPT_PX = 8.0   # XY error must be below this (px) to count as "aligned"

# Low-pass filter coefficients for centroid and step smoothing.
# Values closer to 1 = react faster; closer to 0 = smoother but more lag.
ALPHA_CENTROID = 0.35
ALPHA_STEP_XY  = 0.50

# Proportional gain for XY corrections.
# Automatically reduced as the robot descends (see gain scaling below).
GAIN_XY     = 0.30
GAIN_XY_MIN = 0.05   # gain is never scaled below this floor

MAX_STEP_XY_MM = 0.5   # maximum XY correction per loop cycle (mm) — limits overshoot

# Pixel-to-mm conversion factors.
# How to calibrate: command the robot to move exactly 1 mm in X (or Y),
# measure how many pixels the centroid shifts in the image, then MM_PER_PX = 1 / that_shift.
MM_PER_PX_X = 0.1   # mm per pixel in camera X direction — CALIBRATE BEFORE USE
MM_PER_PX_Y = 0.1   # mm per pixel in camera Y direction — CALIBRATE BEFORE USE

# Camera-to-robot axis mapping.
# If correcting in X moves the robot the wrong direction, set FLIP_X = True.
# If camera X maps to robot Y (rotated mount), set SWAP_XY = True.
FLIP_X  = False
FLIP_Y  = False
SWAP_XY = True

# Number of consecutive frames XY must stay within CENTER_ACCEPT_PX before
# Z approach is allowed to start. Prevents descending onto a briefly-centered flicker.
XY_ALIGN_HOLD_FRAMES = 5


# =============================================================================
# Z CONTROL (area-based)
#
# Detected pixel area grows as the robot gets closer to the object.
# The robot descends until area_px reaches AREA_TARGET.
#
# KEY RULE: Z only moves when XY is already aligned (XY takes priority).
# This prevents XY and Z corrections from fighting each other.
#
# Press 't' at runtime to set AREA_TARGET from the current frame.
# =============================================================================

AREA_TARGET    = 70000.0
AREA_TOLERANCE =  1000.0
GAIN_Z_AREA    = 0.00008
ALPHA_STEP_Z   =  0.35
MAX_STEP_Z_MM  =  0.35
FLIP_Z         = False


# =============================================================================
# ROBOT WRAPPER
# Wraps the Mecademic API so the rest of the code doesn't need to know about
# API quirks (e.g. different pose-read method names across firmware versions).
# =============================================================================

class Meca500Client:
    def __init__(self, ip: str, dry_run: bool = True):
        self.ip      = ip
        self.dry_run = dry_run
        self.robot   = None

    def _read_pose(self):
        # Different firmware versions expose the pose under different method names.
        # Try each in order until one succeeds.
        if self.robot is None:
            return None
        for method_name in ("GetRtTargetCartPos", "GetRtCartPos", "GetPose"):
            try:
                val = getattr(self.robot, method_name)()
                if val is not None:
                    return val
            except Exception:
                pass
        return None

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

    def move_zero_joints(self):
        if self.dry_run:
            print(f"[Robot] DRY_RUN MoveJoints{ZERO_JOINTS}")
            return
        self.robot.SetBlending(0)
        self.robot.SetJointVelLimit(JOINT_VEL_LIMIT)
        self.robot.SetCartLinVel(TRAVEL_CART_LIN_VEL)
        self.robot.SetCartAcc(TRAVEL_CART_ACC)
        self.robot.ResumeMotion()
        self.robot.MoveJoints(*ZERO_JOINTS)
        self.robot.WaitIdle()
        print("[Robot] At zero joints.")

    def move_to_search_pose(self):
        # Keep current XY, only change Z and orientation.
        if self.dry_run:
            print(f"[Robot] DRY_RUN → search pose Z={SEARCH_Z}")
            return
        pose = self._read_pose()
        if pose is None or len(pose) < 6:
            raise RuntimeError("[Robot] Cannot read current pose.")
        x, y = float(pose[0]), float(pose[1])
        self.robot.SetBlending(0)
        self.robot.SetCartLinVel(APPROACH_CART_LIN_VEL)
        self.robot.SetCartAcc(APPROACH_CART_ACC)
        self.robot.ResumeMotion()
        self.robot.MovePose(x, y, SEARCH_Z, TRACK_ALPHA, TRACK_BETA, TRACK_GAMMA)
        self.robot.WaitIdle()
        print(f"[Robot] At search pose Z={SEARCH_Z}.")

    def _set_track_speed(self):
        self.robot.SetBlending(TRACK_BLENDING)
        self.robot.SetCartLinVel(TRACK_CART_LIN_VEL)
        self.robot.SetCartAcc(TRACK_CART_ACC)
        self.robot.ResumeMotion()

    def move_xy_rel(self, dx_mm: float, dy_mm: float):
        # WaitIdle is intentional here: without it, servo moves queue up inside the
        # robot faster than they execute. When the robot catches up it lurches through
        # a pile of stale corrections. With WaitIdle + high TRACK_CART_ACC each step
        # finishes within ~DT, keeping the loop in sync with the robot's real position.
        dx_mm = float(np.clip(dx_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))
        dy_mm = float(np.clip(dy_mm, -MAX_STEP_XY_MM, MAX_STEP_XY_MM))

        if self.dry_run:
            print(f"[Robot] DRY_RUN XY  dX={dx_mm:+.3f}  dY={dy_mm:+.3f}")
            return

        self._set_track_speed()
        if SWAP_XY:
            self.robot.MoveLinRelWrf(dy_mm, dx_mm, 0.0, 0.0, 0.0, 0.0)
        else:
            self.robot.MoveLinRelWrf(dx_mm, dy_mm, 0.0, 0.0, 0.0, 0.0)
        self.robot.WaitIdle()

    def move_z_rel(self, dz_mm: float):
        # Clamp and check safety floor before every downward move.
        dz_mm = float(np.clip(dz_mm, -MAX_STEP_Z_MM, MAX_STEP_Z_MM))

        if self.dry_run:
            print(f"[Robot] DRY_RUN Z   dZ={dz_mm:+.3f}")
            return

        if dz_mm < 0.0:
            pose = self._read_pose()
            if pose is not None and len(pose) >= 3:
                if float(pose[2]) + dz_mm < MIN_Z_MM:
                    print(f"[Robot] MIN_Z={MIN_Z_MM} mm — Z move blocked.")
                    return

        self._set_track_speed()
        self.robot.MoveLinRelWrf(0.0, 0.0, float(dz_mm), 0.0, 0.0, 0.0)
        self.robot.WaitIdle()

    def move_z_jog(self, dz_mm: float):
        # Manual jog from keyboard — same safety check but uses the slower approach speed.
        if self.dry_run:
            print(f"[Robot] DRY_RUN jog  dZ={dz_mm:+.3f}")
            return
        pose = self._read_pose()
        if pose is not None and len(pose) >= 3:
            if float(pose[2]) + dz_mm < MIN_Z_MM:
                print(f"[Robot] MIN_Z={MIN_Z_MM} mm — jog blocked.")
                return
        self.robot.SetBlending(0)
        self.robot.SetCartLinVel(APPROACH_CART_LIN_VEL)
        self.robot.SetCartAcc(APPROACH_CART_ACC)
        self.robot.ResumeMotion()
        self.robot.MoveLinRelWrf(0.0, 0.0, float(dz_mm), 0.0, 0.0, 0.0)
        self.robot.WaitIdle()
        print(f"[Robot] Jog Z  dZ={dz_mm:+.3f}")

    def close(self):
        if self.dry_run or self.robot is None:
            return
        try:
            self.robot.DeactivateRobot()
            self.robot.Disconnect()
        except Exception:
            pass


# =============================================================================
# CAMERA / FILE HELPERS
# =============================================================================

def open_camera():
    backend = cv2.CAP_DSHOW if USE_DSHOW else 0
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    return cap


def save_picture(frame):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = SAVE_DIR / f"image_{time.strftime('%Y%m%d_%H%M%S')}.png"
    if cv2.imwrite(str(filename), frame):
        print(f"[Camera] Saved: {filename}")
    else:
        print("[Camera] Failed to save picture.")


def start_video_writer(frame_shape, fps: float = 10.0):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    filename = SAVE_DIR / f"video_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    h, w     = frame_shape[:2]
    writer   = cv2.VideoWriter(str(filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print("[Video] Failed to start recording.")
        return None, None
    print(f"[Video] Recording started: {filename}")
    return writer, filename


def stop_video_writer(writer, filename):
    if writer is not None:
        writer.release()
        print(f"[Video] Saved: {filename}")


# =============================================================================
# INTENSITY-BASED DETECTION
# =============================================================================

def detect_object(frame_bgr):
    # Detect the brightest object whose grayscale value falls in [I_MIN, I_MAX].
    # Returns (contour, centroid, area_px, radius_px, bounding_box, mask).
    # Returns (None, None, area_px, None, None, mask) if nothing qualifies.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.inRange(blur, I_MIN, I_MAX)

    # Open removes small noise specks; close fills small holes inside the blob.
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area_px     = float(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or area_px < MIN_CONTOUR_AREA:
        return None, None, area_px, None, None, mask

    best = max(contours, key=cv2.contourArea)
    M    = cv2.moments(best)
    if M["m00"] == 0:
        return None, None, area_px, None, None, mask

    cx, cy   = M["m10"] / M["m00"], M["m01"] / M["m00"]
    radius   = float(np.sqrt(area_px / np.pi))   # equivalent circle radius (for display)
    bbox     = cv2.boundingRect(best)
    return best, np.array([cx, cy], dtype=float), area_px, radius, bbox, mask


# =============================================================================
# MAIN
# =============================================================================

def main():
    # ------------------------------------------------------------------
    # State machine:
    #   WAIT_DETECT  — no object visible, robot stays still
    #   ALIGN_XY     — object found, centering XY only (Z locked)
    #   APPROACH_Z   — XY aligned, descending until area reaches target
    #                  Z only moves when XY is within CENTER_ACCEPT_PX
    #   HOLD         — area target reached, maintain XY only (Z locked)
    # ------------------------------------------------------------------
    state         = "WAIT_DETECT"
    align_counter = 0       # frames in a row where XY was within CENTER_ACCEPT_PX
    area_target   = AREA_TARGET
    area_ref      = 0.0     # area snapshot when APPROACH_Z starts — used for gain scaling

    centroid_f   = None        # low-pass filtered centroid
    step_xy_f    = np.zeros(2, dtype=float)
    step_z_f     = 0.0
    next_time    = time.time()

    robot        = Meca500Client(ROBOT_IP, dry_run=DRY_RUN)
    cap          = None
    video_writer = None
    video_file   = None
    is_recording = False

    try:
        # ---- Hardware init ----
        robot.connect()
        robot.move_zero_joints()
        print(f"[Main] Waiting {DELAY_AFTER_ZERO_S:.1f} s...")
        time.sleep(DELAY_AFTER_ZERO_S)
        robot.move_to_search_pose()

        # ---- Camera init ----
        cap = open_camera()
        ret, frame0 = cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera.")
        h0, w0    = frame0.shape[:2]
        target_px = np.array([w0 / 2.0, h0 / 2.0], dtype=float)   # image center = robot aim point

        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask",   cv2.WINDOW_NORMAL)

        print("\n=== Running ===")
        print("Flow:  WAIT_DETECT → ALIGN_XY → APPROACH_Z → HOLD")
        print("Keys:  t=set area target | z=+Z | s=-Z | p=picture | v=video | q=quit\n")

        while True:
            # ---- Fixed-rate timing (10 Hz) ----
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            next_time += DT

            # ---- Grab frame ----
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame read failed.")
                break

            display = frame.copy()
            cv2.circle(display,
                       (int(target_px[0]), int(target_px[1])),
                       CAM_CENTER_RADIUS_PX, (0, 0, 0), CAM_CENTER_THICKNESS)

            # ---- Detect ----
            contour, centroid, area_px, radius_px, bbox, mask = detect_object(frame)

            # Reset per-frame computed values
            err_px      = np.zeros(2, dtype=float)
            err_norm    = 0.0
            step_xy_cmd = np.zeros(2, dtype=float)
            step_z_cmd  = 0.0
            gain_scale  = 1.0
            area_error  = area_target - (area_px if area_px else 0.0)

            # ==================================================================
            # OBJECT NOT DETECTED
            # ==================================================================
            if contour is None:
                centroid_f    = None
                step_xy_f[:]  = 0.0
                step_z_f      = 0.0
                align_counter = 0
                if state in ("ALIGN_XY", "APPROACH_Z"):
                    print("[Main] Object lost — back to WAIT_DETECT.")
                    state = "WAIT_DETECT"

            # ==================================================================
            # OBJECT DETECTED
            # ==================================================================
            else:
                # Draw bounding box and contour on display frame
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                x_b, y_b, w_b, h_b = bbox
                cv2.rectangle(display, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 255, 0), 1)

                # Low-pass filter the centroid to suppress pixel noise
                if centroid_f is None:
                    centroid_f = centroid.copy()
                else:
                    centroid_f = (1 - ALPHA_CENTROID) * centroid_f + ALPHA_CENTROID * centroid
                cv2.drawMarker(display,
                               (int(centroid_f[0]), int(centroid_f[1])),
                               (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                # ---- Gain scaling -----------------------------------------------
                # As the robot descends, the object fills more pixels. The same
                # physical mm now spans more pixels, so the same GAIN_XY produces
                # a larger correction → overshoot → oscillation.
                #
                # Fix: scale GAIN_XY down proportionally as area grows.
                # area ∝ 1/Z², so linear pixel scale ∝ 1/Z ∝ sqrt(area_ref/area_px).
                # At search height (area = area_ref): gain_scale = 1.0
                # At 2× area:  gain_scale ≈ 0.71  (71 % of base gain)
                # At 4× area:  gain_scale ≈ 0.50  (50 % of base gain)
                # ----------------------------------------------------------------
                if state in ("APPROACH_Z", "HOLD") and area_ref > 0:
                    gain_scale = float(np.clip(
                        np.sqrt(area_ref / max(area_px, 1.0)),
                        GAIN_XY_MIN / GAIN_XY,   # floor: never drop below GAIN_XY_MIN
                        1.0                       # ceiling: never exceed the base gain
                    ))
                effective_gain_xy = GAIN_XY * gain_scale

                # ---- XY pixel error → smooth mm step ----
                # Use a wider deadband during Z approach to avoid fighting the Z motion
                deadband   = DEADBAND_PX_APPROACH if state == "APPROACH_Z" else DEADBAND_PX
                err_px     = centroid_f - target_px
                err_norm   = float(np.linalg.norm(err_px))

                ctrl_err   = err_px.copy()
                if err_norm < deadband:
                    ctrl_err[:] = 0.0   # inside deadband — treat as zero error

                px_to_mm   = np.array([[MM_PER_PX_X, 0.0], [0.0, MM_PER_PX_Y]])
                step_xy    = -effective_gain_xy * (px_to_mm @ ctrl_err)
                if FLIP_X: step_xy[0] *= -1.0
                if FLIP_Y: step_xy[1] *= -1.0
                step_xy_f   = (1 - ALPHA_STEP_XY) * step_xy_f + ALPHA_STEP_XY * step_xy
                step_xy_cmd = step_xy_f.copy()

                # ---- Area error → smooth Z step ----
                xy_aligned = (err_norm <= CENTER_ACCEPT_PX)

                if abs(area_error) > AREA_TOLERANCE:
                    raw_z    = -GAIN_Z_AREA * area_error
                    if FLIP_Z: raw_z *= -1.0
                    raw_z    = float(np.clip(raw_z, -MAX_STEP_Z_MM, MAX_STEP_Z_MM))
                    step_z_f = (1 - ALPHA_STEP_Z) * step_z_f + ALPHA_STEP_Z * raw_z
                else:
                    step_z_f = 0.0
                step_z_cmd = float(step_z_f)

                # ==================================================================
                # STATE MACHINE
                # ==================================================================
                if state == "WAIT_DETECT":
                    print("[Main] Object detected → ALIGN_XY")
                    state         = "ALIGN_XY"
                    align_counter = 0

                elif state == "ALIGN_XY":
                    if np.linalg.norm(step_xy_cmd) > 1e-9:
                        robot.move_xy_rel(float(step_xy_cmd[0]), float(step_xy_cmd[1]))
                    align_counter = (align_counter + 1) if xy_aligned else 0
                    if align_counter >= XY_ALIGN_HOLD_FRAMES:
                        print(f"[Main] XY aligned ({XY_ALIGN_HOLD_FRAMES} frames) → APPROACH_Z")
                        state         = "APPROACH_Z"
                        area_ref      = area_px   # snapshot area now — used for gain scaling
                        step_z_f      = 0.0
                        align_counter = 0

                elif state == "APPROACH_Z":
                    # Always correct XY first, even during descent
                    if np.linalg.norm(step_xy_cmd) > 1e-9:
                        robot.move_xy_rel(float(step_xy_cmd[0]), float(step_xy_cmd[1]))

                    # Only move Z when XY is already aligned — prevents XY and Z fighting
                    if xy_aligned and abs(step_z_cmd) > 1e-9:
                        robot.move_z_rel(step_z_cmd)

                    if abs(area_error) <= AREA_TOLERANCE:
                        print(f"[Main] Z target reached  area={area_px:.0f} px² → HOLD")
                        state    = "HOLD"
                        step_z_f = 0.0

                elif state == "HOLD":
                    # Z locked — only keep XY centered
                    if np.linalg.norm(step_xy_cmd) > 1e-9:
                        robot.move_xy_rel(float(step_xy_cmd[0]), float(step_xy_cmd[1]))

            # ==================================================================
            # HUD OVERLAY
            # ==================================================================
            state_color = {
                "WAIT_DETECT": (0, 165, 255),    # orange
                "ALIGN_XY":    (0, 255, 255),    # yellow
                "APPROACH_Z":  (255, 165,   0),  # light blue
                "HOLD":        (0, 255,   0),    # green
            }.get(state, (255, 255, 255))

            rec_tag = "  [REC]" if is_recording else ""
            hud_lines = [
                (f"State: {state}{rec_tag}",
                 (0, 0, 255) if is_recording else state_color),
                (f"Area: {area_px:.0f}  Target: {area_target:.0f}  Error: {area_error:+.0f} px²",
                 (0, 255, 255)),
                (f"Gain scale: {gain_scale:.2f}  Effective gain: {GAIN_XY * gain_scale:.3f}"
                 f"  (area_ref={area_ref:.0f})",
                 (0, 255, 255)),
                (f"Centroid: ({centroid_f[0]:.1f}, {centroid_f[1]:.1f})"
                 if centroid_f is not None else "Centroid: --",
                 (0, 255, 255)),
                (f"err_px: ({err_px[0]:+.1f}, {err_px[1]:+.1f})   "
                 f"step_xy: ({step_xy_cmd[0]:+.3f}, {step_xy_cmd[1]:+.3f}) mm",
                 (0, 255, 255)),
                (f"step_z: {step_z_cmd:+.4f} mm   MIN_Z safety: {MIN_Z_MM:.0f} mm",
                 (0, 255, 255)),
                (f"DRY={robot.dry_run} | t=target  z=+Z  s=-Z  p=pic  v=video  q=quit",
                 (180, 180, 180)),
            ]
            for i, (text, color) in enumerate(hud_lines):
                cv2.putText(display, text, (10, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if is_recording and video_writer is not None:
                video_writer.write(display)

            cv2.imshow("Camera", display)
            cv2.imshow("Mask",   mask)

            # ==================================================================
            # KEYBOARD CONTROLS
            # ==================================================================
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("t"):
                # Set the area target to whatever the object looks like right now
                if area_px and area_px >= MIN_CONTOUR_AREA:
                    area_target = area_px
                    print(f"[Target] Area target set to {area_target:.0f} px²")
                    if state == "HOLD":
                        state    = "APPROACH_Z"
                        step_z_f = 0.0
                        print("[Main] New target — returning to APPROACH_Z")
                else:
                    print("[Target] No object visible — target not changed.")

            elif key == ord("z"):
                robot.move_z_jog(+JOG_STEP_MM)

            elif key == ord("s"):
                robot.move_z_jog(-JOG_STEP_MM)

            elif key == ord("p"):
                save_picture(display)

            elif key == ord("v"):
                if not is_recording:
                    video_writer, video_file = start_video_writer(
                        display.shape, fps=max(1.0, 1.0 / DT)
                    )
                    if video_writer is not None:
                        is_recording = True
                else:
                    stop_video_writer(video_writer, video_file)
                    video_writer = None
                    video_file   = None
                    is_recording = False

    finally:
        # Always runs — releases all hardware even if an exception was raised mid-loop
        if video_writer is not None:
            stop_video_writer(video_writer, video_file)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
