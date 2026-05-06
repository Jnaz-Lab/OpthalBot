import time
# import clr   # uncomment when shutter is reconnected
import numpy as np
import mecademicpy.robot as mdr

# Thorlabs DLL references — commented out while shutter is not in use.
# Uncomment all three blocks (clr, AddReference, from...) to re-enable.
# clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
# clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.KCube.SolenoidCLI.dll")
# from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
# from Thorlabs.MotionControl.KCube.SolenoidCLI import KCubeSolenoid, SolenoidStatus


# =============================================================================
# USER SETTINGS — adjust these before each experiment
# =============================================================================

ROBOT_IP = "192.168.0.100"   # Ethernet IP of the Mecademic robot
# SHUTTER_SERIAL = "68000318"

# Tool orientation used for every move.
# (0, 90, 0) = tool pointing straight down.
# Dynamic normal-aligned orientation is implemented but disabled below — see USE_DYNAMIC_NORMAL_ORIENTATION.
FIXED_SAFE_ORIENTATION         = (0, 90, 0)
USE_DYNAMIC_NORMAL_ORIENTATION = False   # set True once 3D positions are validated

# Spherical cap geometry
# The sphere is defined by a center point and radius in the robot frame.
# The cap is the portion of the sphere we want to print on (like the top of a dome).
SPHERE_CENTER       = np.array([190.0, 0.0, 136.0])  # (X, Y, Z) of sphere center in robot frame (mm)
SPHERE_RADIUS_MM    = 20.0   # radius of the sphere (mm)
OUTER_CAP_RADIUS_MM = 20.0   # radius of the outermost ring measured flat (in the XY plane) (mm)
                              # must be <= SPHERE_RADIUS_MM; at 20=20 this covers a full hemisphere

POINT_SPACING_MM = 1.0   # desired arc-length between laser pulses along each ring (mm)
RING_SPACING_MM  = POINT_SPACING_MM   # desired arc-length between rings along the sphere surface (mm)

# STANDOFF_MM: if > 0, the tool tip is offset away from the surface along the local normal.
# Useful to avoid touching the sample. Set to 0 for direct surface contact.
STANDOFF_MM = 0.0

# APPROACH_OFFSET_MM: how far above each deposition point the robot pauses before descending.
# Gives a safe intermediate waypoint to avoid collisions when the surface is curved.
APPROACH_OFFSET_MM = 20.0

# CAP_DIRECTION: which side of the sphere is the cap.
#   +1 = upper hemisphere (cap apex is above sphere center)
#   -1 = lower hemisphere (cap apex is below sphere center)
CAP_DIRECTION = 1

# NORMAL_SIGN: which way the surface normal points.
#   +1 = outward (away from sphere center) — tool approaches from outside
#   -1 = inward (toward sphere center)    — tool approaches from inside
NORMAL_SIGN = 1

SHUTTER_OPEN_TIME_S = 0.05   # laser pulse duration per point (seconds)
START_DELAY_S       = 3.0    # pause at the first point before printing starts (seconds)

# Two separate motion profiles:
# TRAVEL = used when moving between positions (can be faster)
# PRINT  = used during the actual deposition sequence (should be slow and controlled)
TRAVEL_BLENDING        = 0
TRAVEL_CART_ACC        = 25
TRAVEL_CART_ANG_VEL    = 20
TRAVEL_JOINT_VEL_LIMIT = 25
TRAVEL_CART_LIN_VEL    = 20

PRINT_BLENDING         = 0
PRINT_CART_ACC         = 25
PRINT_CART_ANG_VEL     = 20
PRINT_JOINT_VEL_LIMIT  = 25
PRINT_CART_LIN_VEL     = 20


# =============================================================================
# SHUTTER FUNCTIONS — disabled while shutter is not connected
# =============================================================================
# def initialize_shutter(serial_no: str):
#     DeviceManagerCLI.BuildDeviceList()
#     device = KCubeSolenoid.CreateKCubeSolenoid(serial_no)
#     device.Connect(serial_no)
#     device.WaitForSettingsInitialized(5000)
#     device.StartPolling(250)
#     time.sleep(0.25)
#     device.EnableDevice()
#     time.sleep(0.5)
#     device.SetOperatingMode(SolenoidStatus.OperatingModes.Manual)
#     return device
#
# def pulse_shutter(device, pulse_time_s: float):
#     device.SetOperatingState(SolenoidStatus.OperatingStates.Active)
#     time.sleep(pulse_time_s)
#     device.SetOperatingState(SolenoidStatus.OperatingStates.Inactive)
#
# def shutdown_shutter(device):
#     if device is not None:
#         for action in [
#             lambda: device.SetOperatingState(SolenoidStatus.OperatingStates.Inactive),
#             lambda: device.StopPolling(),
#             lambda: device.Disconnect(),
#         ]:
#             try: action()
#             except Exception: pass


# =============================================================================
# ROBOT FUNCTIONS
# =============================================================================

def initialize_robot(ip_address: str):
    robot = mdr.Robot()
    robot.Connect(address=ip_address)
    robot.ActivateRobot()
    robot.Home()
    print("Robot: connected, activated, and homed.")
    return robot


def configure_travel_motion(robot):
    # Apply travel-speed settings (used when repositioning, not printing).
    robot.SetBlending(TRAVEL_BLENDING)
    robot.SetCartAcc(TRAVEL_CART_ACC)
    robot.SetCartAngVel(TRAVEL_CART_ANG_VEL)
    robot.SetJointVelLimit(TRAVEL_JOINT_VEL_LIMIT)
    robot.SetCartLinVel(TRAVEL_CART_LIN_VEL)


def configure_print_motion(robot):
    # Apply print-speed settings (used during the deposition sequence).
    robot.SetBlending(PRINT_BLENDING)
    robot.SetCartAcc(PRINT_CART_ACC)
    robot.SetCartAngVel(PRINT_CART_ANG_VEL)
    robot.SetJointVelLimit(PRINT_JOINT_VEL_LIMIT)
    robot.SetCartLinVel(PRINT_CART_LIN_VEL)


def move_to_start(robot, first_point, first_orientation, first_normal):
    # Three-step approach to avoid collisions when reaching the first print point:
    # 1) Zero joints (known safe posture)
    # 2) Move to a point directly above the sphere center (clears any obstacles near the cap)
    # 3) Move to the approach waypoint above the first print point, then descend to it
    configure_travel_motion(robot)

    print("Moving to zero joints...")
    robot.MoveJoints(0, 0, 0, 0, 0, 0)
    robot.WaitIdle()

    # Safe intermediate height: directly above the sphere center, clear of the cap surface
    safe_z = SPHERE_CENTER[2] + SPHERE_RADIUS_MM + APPROACH_OFFSET_MM
    print("Moving to safe pose above sphere center...")
    robot.MoveLin(SPHERE_CENTER[0], SPHERE_CENTER[1], safe_z, *first_orientation)
    robot.WaitIdle()

    # Approach waypoint: APPROACH_OFFSET_MM above the first deposition point along the surface normal
    approach = first_point + first_normal * APPROACH_OFFSET_MM
    print("Moving to approach waypoint above first print point...")
    robot.MoveLin(*approach, *first_orientation)
    robot.WaitIdle()

    print("Descending to first print point...")
    robot.MoveLin(*first_point, *first_orientation)
    robot.WaitIdle()


def move_to_home(robot, last_point, last_orientation, last_normal):
    # Reverse of move_to_start: retract along the normal first, then go to zero joints.
    # Moving to the cap center pose before retracting keeps the path predictable.
    configure_travel_motion(robot)

    print("Moving to cap center pose...")
    robot.MoveLin(*last_point, *last_orientation)
    robot.WaitIdle()

    # Retract away from the surface before any large joint motion
    retract = last_point + last_normal * APPROACH_OFFSET_MM
    print("Retracting from cap center along the normal...")
    robot.MoveLin(*retract, *last_orientation)
    robot.WaitIdle()

    print("Returning to zero joints...")
    robot.MoveJoints(0, 0, 0, 0, 0, 0)
    robot.WaitIdle()


def shutdown_robot(robot):
    if robot is None:
        return
    for action in [
        lambda: robot.WaitIdle(),
        lambda: robot.DeactivateRobot(),
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

def normalize(v):
    # Return unit vector. Raises if the input is nearly zero-length (would divide by ~0).
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def surface_normal(surface_point, sphere_center, normal_sign=1):
    # The outward normal at any point on a sphere is simply the direction from
    # the center to that point. Multiplying by normal_sign lets you flip it inward.
    return normal_sign * normalize(surface_point - sphere_center)


def rotation_matrix_to_xyz_euler_deg(R):
    # Convert a 3x3 rotation matrix to intrinsic XYZ Euler angles (degrees).
    # This matches the Mecademic alpha-beta-gamma convention: R = Rx(a) @ Ry(b) @ Rz(g).
    # NOTE: if the physical tool mounting differs from the assumed axis, adjust this function.
    beta = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    cos_beta = np.cos(beta)

    if abs(cos_beta) > 1e-9:
        alpha = np.arctan2(-R[1, 2], R[2, 2])
        gamma = np.arctan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock: beta is ±90°, so alpha and gamma are coupled.
        # Assign all rotation to gamma and set alpha=0 as a standard fallback.
        alpha = 0.0
        gamma = np.arctan2(R[1, 0], R[1, 1])

    return np.degrees([alpha, beta, gamma])


def tool_orientation_from_normal(normal_world):
    # Returns the tool orientation (alpha, beta, gamma) needed to align the tool axis
    # with the surface normal.
    # Currently returns a fixed known-good orientation while 3D positions are being validated.
    # Once validated, replace with a proper normal-to-Euler conversion here.
    return np.array(FIXED_SAFE_ORIENTATION, dtype=float)


def sphere_point(theta, phi, sphere_center, sphere_radius, cap_direction=1):
    # Compute a 3D point on the sphere using spherical coordinates.
    # theta = polar angle from the cap apex (0 = apex, increases outward toward the equator)
    # phi   = azimuthal angle around the cap axis (0 to 2π)
    #
    # ring_radius = the flat (XY-plane) radius of the ring at this theta
    # z_offset    = signed height above/below sphere center (sign depends on CAP_DIRECTION)
    cx, cy, cz = sphere_center
    ring_radius = sphere_radius * np.sin(theta)
    z_offset    = cap_direction * sphere_radius * np.cos(theta)

    return np.array([
        cx + ring_radius * np.cos(phi),
        cy + ring_radius * np.sin(phi),
        cz + z_offset,
    ])


def compute_ring_thetas(sphere_radius, outer_cap_radius, ring_spacing):
    # Calculate the polar angles (theta) for each ring, working outward to inward.
    #
    # theta_max = polar angle of the outermost ring (derived from its flat radius)
    # Rings are spaced equally in arc-length along the sphere surface.
    # Returns rings from outermost (largest theta) to innermost (smallest theta).
    if outer_cap_radius <= 0:
        return np.array([]), 0, 0.0
    if outer_cap_radius > sphere_radius:
        raise ValueError("OUTER_CAP_RADIUS_MM must be <= SPHERE_RADIUS_MM.")

    theta_max    = np.arcsin(outer_cap_radius / sphere_radius)
    arc_to_edge  = sphere_radius * theta_max   # total arc length from apex to outer edge

    num_rings          = max(1, round(arc_to_edge / ring_spacing))
    actual_ring_spacing = arc_to_edge / num_rings

    print(f"Desired ring spacing  = {ring_spacing:.3f} mm")
    print(f"Number of rings       = {num_rings}")
    print(f"Actual ring spacing   = {actual_ring_spacing:.3f} mm")

    # Build thetas from innermost to outermost, then reverse so we print outer-first.
    thetas = np.linspace(theta_max / num_rings, theta_max, num_rings)[::-1]
    return thetas, num_rings, actual_ring_spacing


def generate_ring_points(theta, sphere_center, sphere_radius, point_spacing, cap_direction=1):
    # Compute all 3D points on one ring at polar angle theta.
    # Number of points is chosen so the arc-length spacing is close to point_spacing.
    ring_radius   = sphere_radius * np.sin(theta)
    circumference = 2 * np.pi * ring_radius

    num_points     = max(3, round(circumference / point_spacing))
    actual_spacing = circumference / num_points

    print(
        f"  Ring theta={np.degrees(theta):.2f}°  ring radius={ring_radius:.3f} mm  "
        f"→ {num_points} points, actual spacing={actual_spacing:.3f} mm"
    )

    # Drop the last phi (= 2π = same as 0) to avoid printing the same spot twice.
    phi_values = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]

    return [
        sphere_point(theta, phi, sphere_center, sphere_radius, cap_direction)
        for phi in phi_values
    ]


def deposition_pose(surface_point, sphere_center, standoff_mm=0.0, normal_sign=1):
    # Given a raw point on the sphere surface, compute:
    #   - the actual tool position (offset by standoff_mm along the normal if > 0)
    #   - the tool orientation aligned to the surface normal
    #   - the surface normal itself (needed later to compute approach waypoints)
    normal      = surface_normal(surface_point, sphere_center, normal_sign)
    tool_point  = surface_point + normal * standoff_mm
    orientation = tool_orientation_from_normal(normal)
    return tool_point, orientation, normal


def approach_point(deposition_pt, normal, offset_mm):
    # Waypoint used before descending to a deposition point.
    # Located offset_mm above the deposition point along the surface normal.
    return deposition_pt + normal * offset_mm


def first_ring_pose():
    # Compute the pose of the very first print point:
    # the first point (phi=0) on the outermost ring.
    # Called before motion starts to know where to move the robot.
    thetas, _, _ = compute_ring_thetas(SPHERE_RADIUS_MM, OUTER_CAP_RADIUS_MM, RING_SPACING_MM)
    first_surface_pt = sphere_point(thetas[0], 0.0, SPHERE_CENTER, SPHERE_RADIUS_MM, CAP_DIRECTION)
    return deposition_pose(first_surface_pt, SPHERE_CENTER, STANDOFF_MM, NORMAL_SIGN)


def apex_pose():
    # Compute the pose at the cap apex (theta=0, the very top of the dome).
    # This is the final deposition point and also used as the home-return reference.
    apex_surface_pt = sphere_point(0.0, 0.0, SPHERE_CENTER, SPHERE_RADIUS_MM, CAP_DIRECTION)
    return deposition_pose(apex_surface_pt, SPHERE_CENTER, STANDOFF_MM, NORMAL_SIGN)


# =============================================================================
# MAIN PROCESS
# =============================================================================

def print_hemisphere_rings(robot, device):
    # Move through every ring from outermost to innermost, firing the laser at each point.
    # Finishes with one pulse at the cap apex (theta=0) which is not part of any ring.
    configure_print_motion(robot)

    thetas, num_rings, _ = compute_ring_thetas(SPHERE_RADIUS_MM, OUTER_CAP_RADIUS_MM, RING_SPACING_MM)

    for ring_idx, theta in enumerate(thetas, start=1):
        print(f"\nRing {ring_idx}/{num_rings}  (theta = {np.degrees(theta):.2f}°)")

        points = generate_ring_points(theta, SPHERE_CENTER, SPHERE_RADIUS_MM, POINT_SPACING_MM, CAP_DIRECTION)

        for pt_idx, surface_pt in enumerate(points, start=1):
            tool_pt, orientation, normal = deposition_pose(surface_pt, SPHERE_CENTER, STANDOFF_MM, NORMAL_SIGN)

            print(
                f"    Point {pt_idx}/{len(points)}  "
                f"X={tool_pt[0]:.3f}  Y={tool_pt[1]:.3f}  Z={tool_pt[2]:.3f}  "
                f"A={orientation[0]:.1f}  B={orientation[1]:.1f}  G={orientation[2]:.1f}"
            )

            robot.MoveLin(*tool_pt, *orientation)
            robot.WaitIdle()

            # pulse_shutter(device, SHUTTER_OPEN_TIME_S)   # uncomment when shutter is reconnected

    # Apex point (center of the cap) — not covered by any ring, printed separately
    apex_pt, apex_orientation, apex_normal = apex_pose()
    print("\nMoving to cap apex (final point)...")
    robot.MoveLin(*apex_pt, *apex_orientation)
    robot.WaitIdle()
    # pulse_shutter(device, SHUTTER_OPEN_TIME_S)

    # Return the apex pose so move_to_home knows where to retract from
    return apex_pt, apex_orientation, apex_normal


def main():
    robot  = None
    device = None

    try:
        # 1 — Connect to hardware (shutter disabled for now)
        # device = initialize_shutter(SHUTTER_SERIAL)
        robot = initialize_robot(ROBOT_IP)

        # 2 — Set reference frames with no offset (coordinates are in the robot's own frame)
        robot.SetWrf(0, 0, 0, 0, 0, 0)
        robot.SetTrf(0, 0, 0, 0, 0, 0)

        # 3 — Compute the first print pose and move to it safely
        first_pt, first_orientation, first_normal = first_ring_pose()
        move_to_start(robot, first_pt, first_orientation, first_normal)

        # 4 — Short pause at the first point (useful to verify position before printing)
        print(f"Waiting {START_DELAY_S:.1f} s at first print position...")
        time.sleep(START_DELAY_S)

        # 5 — Print all rings from outer to inner, ending at the apex
        apex_pt, apex_orientation, apex_normal = print_hemisphere_rings(robot, device)

        # 6 — Retract from the apex and return to home
        move_to_home(robot, apex_pt, apex_orientation, apex_normal)

    finally:
        # Always runs — ensures robot is never left powered on after an error
        # shutdown_shutter(device)
        shutdown_robot(robot)


if __name__ == "__main__":
    main()
