import cv2
import time
import numpy as np
from pathlib import Path

# =============================================================================
# USER SETTINGS
# =============================================================================

ROBOT_IP  = "192.168.0.100"
CAM_INDEX = 0        # 0 = first USB camera
USE_DSHOW = True     # DirectShow backend — more stable on Windows
DRY_RUN   = False    # True = print robot commands without moving (safe testing mode)

ZERO_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

JOINT_VEL_LIMIT = 25
CART_LIN_VEL    = 400
CART_ACC        = 250

DELAY_AFTER_ZERO_S = 1.0

JOG_STEP_MM = 1.0   # how far each keypress moves the robot (mm)

# Pictures are saved next to this script in a "pictures" subfolder
SAVE_DIR = Path(__file__).resolve().parent / "pictures"

# HSV range for blue object detection.
# Tune LOWER_BLUE / UPPER_BLUE if the object isn't being picked up correctly.
# Hue 90–140 covers blue; saturation/value floors filter out grey and dark pixels.
LOWER_BLUE = np.array([ 90,  70,  40], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA = 200   # blobs smaller than this (px²) are ignored as noise


# =============================================================================
# ROBOT WRAPPER
# =============================================================================

class Meca500Client:
    def __init__(self, ip: str, dry_run: bool = True):
        self.ip      = ip
        self.dry_run = dry_run
        self.robot   = None

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
        self.robot.SetJointVelLimit(JOINT_VEL_LIMIT)
        self.robot.SetCartLinVel(CART_LIN_VEL)
        self.robot.SetCartAcc(CART_ACC)
        self.robot.ResumeMotion()
        self.robot.MoveJoints(*ZERO_JOINTS)
        self.robot.WaitIdle()
        print("[Robot] At zero joints.")

    def jog(self, dx: float, dy: float, dz: float):
        # Move the robot by (dx, dy, dz) mm relative to the world reference frame.
        if self.dry_run:
            print(f"[Robot] DRY_RUN jog  dX={dx:+.3f}  dY={dy:+.3f}  dZ={dz:+.3f}")
            return
        self.robot.SetCartLinVel(CART_LIN_VEL)
        self.robot.SetCartAcc(CART_ACC)
        self.robot.ResumeMotion()
        self.robot.MoveLinRelWrf(float(dx), float(dy), float(dz), 0.0, 0.0, 0.0)
        self.robot.WaitIdle()
        print(f"[Robot] Jog  dX={dx:+.3f}  dY={dy:+.3f}  dZ={dz:+.3f}")

    def close(self):
        if self.dry_run or self.robot is None:
            return
        try:
            self.robot.DeactivateRobot()
            self.robot.Disconnect()
        except Exception:
            pass


# =============================================================================
# CAMERA HELPER
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


# =============================================================================
# BLUE OBJECT DETECTION
# =============================================================================

def detect_blue_object(frame_bgr):
    # Convert to HSV and threshold to isolate blue pixels.
    # Returns (contour, center_px, area_px, radius_px, bbox, mask).
    # Returns (None, None, None, None, None, mask) if nothing qualifies.
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.inRange(blur, LOWER_BLUE, UPPER_BLUE)

    # Open removes small noise specks; close fills small holes inside the blob.
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    area_px     = float(cv2.countNonZero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or area_px < MIN_CONTOUR_AREA:
        return None, None, None, None, None, mask

    best = max(contours, key=cv2.contourArea)
    M    = cv2.moments(best)
    center = None
    if M["m00"] != 0:
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    bbox     = cv2.boundingRect(best)
    # Equivalent circle radius: how big a circle would have this same pixel area.
    # Used as a distance proxy — larger radius = robot is closer to the object.
    radius_px = float(np.sqrt(area_px / np.pi))

    return best, center, area_px, radius_px, bbox, mask


# =============================================================================
# MAIN
# =============================================================================

def main():
    robot = Meca500Client(ROBOT_IP, dry_run=DRY_RUN)
    cap   = None

    # ref_radius_px: set by pressing 'r' — stores the object's apparent radius
    # at the current robot position. Useful for calibrating MM_PER_PX:
    # jog the robot a known distance, note how many pixels the radius changes.
    ref_radius_px = None

    try:
        robot.connect()
        robot.move_zero_joints()

        print(f"[Main] Waiting {DELAY_AFTER_ZERO_S:.1f} s...")
        time.sleep(DELAY_AFTER_ZERO_S)

        cap = open_camera()
        print("\n=== Camera ON ===")
        print("Keys:  x=+X  a=-X  y=+Y  h=-Y  z=+Z  s=-Z  r=save ref radius  p=picture  q=quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Failed to read frame.")
                break

            display = frame.copy()
            img_h, img_w = display.shape[:2]

            # Crosshair at image center
            cv2.circle(display, (img_w // 2, img_h // 2), 10, (0, 0, 0), 2)

            contour, center, area_px, radius_px, bbox, mask = detect_blue_object(frame)

            # Build HUD lines — filled in depending on whether the object is visible
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                if center is not None:
                    cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                x_b, y_b, w_b, h_b = bbox
                cv2.rectangle(display, (x_b, y_b), (x_b + w_b, y_b + h_b), (255, 255, 0), 1)

                ref_str = f"{ref_radius_px:.2f} px" if ref_radius_px is not None else "not set"
                hud = [
                    ("Blue object detected",                           (0, 255, 0)),
                    (f"Center: {center}",                              (0, 255, 255)),
                    (f"Area (all blue px): {area_px:.1f} px²",         (0, 255, 255)),
                    (f"Equivalent radius: {radius_px:.2f} px",         (0, 255, 255)),
                    (f"Reference radius (r): {ref_str}",               (0, 255, 255)),
                    (f"Box: x={x_b}  y={y_b}  w={w_b}  h={h_b}",     (0, 255, 255)),
                ]
            else:
                ref_str = f"{ref_radius_px:.2f} px" if ref_radius_px is not None else "not set"
                hud = [
                    ("Blue object not detected",                       (0, 0, 255)),
                    (f"Reference radius (r): {ref_str}",               (0, 255, 255)),
                ]

            for i, (text, color) in enumerate(hud):
                cv2.putText(display, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Key guide pinned to the bottom of the frame
            cv2.putText(display,
                        "x:+X  a:-X  y:+Y  h:-Y  z:+Z  s:-Z  r:ref  p:pic  q:quit",
                        (10, img_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("Camera",    display)
            cv2.imshow("Blue Mask", mask)

            key = cv2.waitKey(1) & 0xFF

            if   key == ord("q"): break
            elif key == ord("x"): robot.jog(+JOG_STEP_MM,  0.0,          0.0)
            elif key == ord("a"): robot.jog(-JOG_STEP_MM,  0.0,          0.0)
            elif key == ord("y"): robot.jog( 0.0,          +JOG_STEP_MM, 0.0)
            elif key == ord("h"): robot.jog( 0.0,          -JOG_STEP_MM, 0.0)
            elif key == ord("z"): robot.jog( 0.0,           0.0,         +JOG_STEP_MM)
            elif key == ord("s"): robot.jog( 0.0,           0.0,         -JOG_STEP_MM)
            elif key == ord("p"): save_picture(display)
            elif key == ord("r"):
                if contour is not None:
                    ref_radius_px = radius_px
                    print(f"[Ref] Reference radius saved: {ref_radius_px:.2f} px")
                else:
                    print("[Ref] No blue object visible — reference not saved.")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    main()
