import time
import clr
import numpy as np
import mecademicpy.robot as mdr

# Thorlabs Kinesis DLL references — needed to control the laser shutter.
# These files are installed with the Thorlabs Kinesis software on this PC.
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.SolenoidCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.SolenoidCLI import KCubeSolenoid, SolenoidStatus


# =============================================================================
# USER SETTINGS — adjust these before each experiment
# =============================================================================

ROBOT_IP       = "192.168.0.100"   # Ethernet IP of the Mecademic robot
SHUTTER_SERIAL = "68000318"        # Serial number on the KCube solenoid (shutter) box

# Working heights in mm (robot Z axis).
# PRINT_Z  = height at which the laser fires on the sample surface.
# SAFE_Z   = raised height used when moving between positions (avoids dragging the tool).
PRINT_Z = 166.85
SAFE_Z  = PRINT_Z + 30.0

# Tool orientation (alpha, beta, gamma in degrees) — stays constant throughout.
# (0, 90, 0) points the tool straight downward.
TOOL_ORIENTATION = (0, 90, 0)

# Circle pattern parameters
ARC_LENGTH_MM    = 0.5                       # Target spacing between laser pulses along the arc (mm)
OUTER_RADIUS_MM  = 9.0                       # Radius of the outermost circle (mm)
NUM_CIRCLES      = 6                         # How many concentric circles to print
CIRCLE_CENTER_XY = np.array([223.38, 0.0])  # XY center of all circles in the robot frame (mm)

SHUTTER_OPEN_TIME_S = 0.05   # How long to keep the shutter open for each laser pulse (seconds)

# Robot motion parameters — lower values = slower and smoother movement
BLENDING        = 0    # 0 = robot stops fully at each point before moving to the next
CART_ACC        = 20   # Cartesian acceleration (mm/s²)
CART_ANG_VEL    = 15   # Cartesian angular velocity (deg/s)
JOINT_VEL_LIMIT = 10   # Joint velocity cap (percentage of max)
CART_LIN_VEL    = 50   # Cartesian linear velocity (mm/s)


# =============================================================================
# SHUTTER FUNCTIONS
# =============================================================================

def initialize_shutter(serial_no: str):
    # Scan for all connected Thorlabs devices, then open the one matching serial_no.
    DeviceManagerCLI.BuildDeviceList()

    device = KCubeSolenoid.CreateKCubeSolenoid(serial_no)
    device.Connect(serial_no)
    device.WaitForSettingsInitialized(5000)  # wait up to 5 s for the device to load its settings

    device.StartPolling(250)  # check device status every 250 ms
    time.sleep(0.25)

    device.EnableDevice()
    time.sleep(0.5)

    # Manual mode: we open and close the shutter ourselves in code (not timed by the device).
    device.SetOperatingMode(SolenoidStatus.OperatingModes.Manual)
    return device


def pulse_shutter(device, pulse_time_s: float):
    # Fire the laser: open shutter → wait → close shutter. One call = one exposure.
    device.SetOperatingState(SolenoidStatus.OperatingStates.Active)
    time.sleep(pulse_time_s)
    device.SetOperatingState(SolenoidStatus.OperatingStates.Inactive)


def shutdown_shutter(device):
    # Safely close and disconnect the shutter.
    # Each step is in a separate try/except so one failure doesn't skip the rest.
    if device is None:
        return
    for action in [
        lambda: device.SetOperatingState(SolenoidStatus.OperatingStates.Inactive),
        lambda: device.StopPolling(),
        lambda: device.Disconnect(),
    ]:
        try:
            action()
        except Exception:
            pass


# =============================================================================
# ROBOT FUNCTIONS
# =============================================================================

def initialize_robot(ip_address: str):
    # Connect over Ethernet, activate the robot (powers the joints),
    # then home it so all coordinates are relative to the calibrated zero position.
    robot = mdr.Robot()
    robot.Connect(address=ip_address)
    robot.ActivateRobot()
    robot.Home()
    print("Robot: connected, activated, and homed.")
    return robot


def configure_robot_motion(robot):
    # Set all motion limits in one place — easy to find and tweak later.
    robot.SetBlending(BLENDING)
    robot.SetCartAcc(CART_ACC)
    robot.SetCartAngVel(CART_ANG_VEL)
    robot.SetJointVelLimit(JOINT_VEL_LIMIT)
    robot.SetCartLinVel(CART_LIN_VEL)


def move_to_start(robot, center_xy):
    # Go to all-zero joints first (a known safe posture) before moving to the print area.
    # This avoids unpredictable paths if the robot was left in an unknown position.
    robot.MoveJoints(0, 0, 0, 0, 0, 0)
    robot.WaitIdle()

    # Move linearly to the center of the circle pattern at the working height.
    robot.MoveLin(center_xy[0], center_xy[1], PRINT_Z, *TOOL_ORIENTATION)
    robot.WaitIdle()


def move_to_home(robot, center_xy):
    # Lift the tool to SAFE_Z before going home so it doesn't drag across the sample.
    robot.MoveLin(center_xy[0], center_xy[1], SAFE_Z, *TOOL_ORIENTATION)
    robot.WaitIdle()

    robot.MoveJoints(0, 0, 0, 0, 0, 0)
    robot.WaitIdle()


def shutdown_robot(robot):
    if robot is None:
        return
    for action in [
        lambda: robot.WaitIdle(),
        lambda: robot.DeactivateRobot(),   # removes power from joints
        lambda: robot.WaitDeactivated(),
        lambda: robot.Disconnect(),
    ]:
        try:
            action()
        except Exception:
            pass
    print("Robot: deactivated and disconnected.")


# =============================================================================
# GEOMETRY FUNCTIONS
# =============================================================================

def generate_circle_points(radius_mm: float, arc_length_mm: float, center_xy: np.ndarray):
    # Calculate how many equally-spaced points fit around the circle
    # so that the gap between consecutive points is close to arc_length_mm.
    circumference = 2 * np.pi * radius_mm
    num_points = max(3, round(circumference / arc_length_mm))  # at least 3 points

    actual_spacing = circumference / num_points
    print(f"  Radius {radius_mm:.2f} mm → {num_points} points, actual spacing {actual_spacing:.3f} mm")

    # Generate angles 0 → 2π, but drop the last angle because it equals 0 (same point twice).
    theta = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]

    x = center_xy[0] + radius_mm * np.cos(theta)
    y = center_xy[1] + radius_mm * np.sin(theta)

    return np.column_stack((x, y))  # shape: (num_points, 2)


def generate_circle_radii(outer_radius_mm: float, num_circles: int):
    # Create evenly-spaced radii from the outer circle inward.
    # The innermost radius = outer_radius / num_circles (so it never reaches zero).
    # Example: outer=9, num=6 → [9.0, 7.2, 5.4, 3.6, 1.8]  (approx)
    if num_circles <= 0:
        return []
    return np.linspace(outer_radius_mm, outer_radius_mm / num_circles, num_circles)


# =============================================================================
# MAIN PROCESS
# =============================================================================

def print_concentric_circles(robot, device, arc_length_mm, outer_radius_mm, center_xy, num_circles=1):
    # Outer loop: iterate over each concentric circle (largest to smallest radius).
    # Inner loop: move to each point on that circle and fire one laser pulse.
    radii = generate_circle_radii(outer_radius_mm, num_circles)

    for circle_idx, radius_mm in enumerate(radii, start=1):
        print(f"\nCircle {circle_idx}/{num_circles}  (radius = {radius_mm:.3f} mm)")

        points = generate_circle_points(radius_mm, arc_length_mm, center_xy)

        for pt_idx, (x, y) in enumerate(points, start=1):
            print(f"    Point {pt_idx}/{len(points)}  X={x:.3f}  Y={y:.3f}")

            robot.MoveLin(x, y, PRINT_Z, *TOOL_ORIENTATION)
            robot.WaitIdle()

            pulse_shutter(device, SHUTTER_OPEN_TIME_S)


def main():
    robot  = None
    device = None

    try:
        # 1 — Connect to the laser shutter and the robot
        device = initialize_shutter(SHUTTER_SERIAL)
        robot  = initialize_robot(ROBOT_IP)

        # 2 — Set reference frames to the robot base and tool flange with no offset.
        #     (0,0,0,0,0,0) means our coordinates are directly in the robot's own frame.
        robot.SetWrf(0, 0, 0, 0, 0, 0)
        robot.SetTrf(0, 0, 0, 0, 0, 0)

        # 3 — Apply motion limits
        configure_robot_motion(robot)

        # 4 — Move to the first print position (center of the circle pattern)
        move_to_start(robot, CIRCLE_CENTER_XY)

        # 5 — Execute the concentric circle laser pattern
        print_concentric_circles(
            robot           = robot,
            device          = device,
            arc_length_mm   = ARC_LENGTH_MM,
            outer_radius_mm = OUTER_RADIUS_MM,
            center_xy       = CIRCLE_CENTER_XY,
            num_circles     = NUM_CIRCLES,
        )

        # 6 — Retract tool and return to home position
        move_to_home(robot, CIRCLE_CENTER_XY)

    finally:
        # This block always runs, even if an error occurs mid-experiment.
        # Ensures the laser shutter and robot are never left in an active state.
        shutdown_shutter(device)
        shutdown_robot(robot)


if __name__ == "__main__":
    main()
