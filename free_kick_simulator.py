# free_kick_simulator.py
import numpy as np
from dataclasses import dataclass

@dataclass
class BallParams:
    mass: float = 0.43
    radius: float = 0.11
    rho_air: float = 1.225
    Cd: float = 0.28
    A: float = np.pi * (0.11**2)

@dataclass
class KickParams:
    speed: float
    azimuth_deg: float
    elevation_deg: float
    spin_rpm: float
    spin_axis_azimuth_deg: float
    spin_axis_elevation_deg: float

@dataclass
class EnvParams:
    ground_z: float = 0.0
    wind_xyz: tuple = (0.0, 0.0, 0.0)
    g: float = 9.81
    goal_x: float = 24.0
    goal_width: float = 7.32
    goal_height: float = 2.44

def rpm_to_rad_per_s(rpm: float) -> float:
    return rpm * 2*np.pi / 60.0

def angle_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return np.array([x, y, z], dtype=float)

def initial_velocity(speed: float, az_deg: float, el_deg: float) -> np.ndarray:
    return speed * angle_to_unit(az_deg, el_deg)

def lift_coefficient(omega_vec: np.ndarray, v_vec: np.ndarray, r: float) -> float:
    v = np.linalg.norm(v_vec)
    if v < 1e-6:
        return 0.0
    S = r * np.linalg.norm(omega_vec) / v
    Cl = (1.6 * S) / (1 + 3.5 * S)
    return float(np.clip(Cl, 0.0, 0.35))

def forces(ball: BallParams, vel_ball: np.ndarray, omega_vec: np.ndarray, env) -> np.ndarray:
    v_rel = vel_ball - np.array(env.wind_xyz, dtype=float)
    v = np.linalg.norm(v_rel)
    Fd = -0.5 * ball.rho_air * ball.Cd * ball.A * v * v_rel
    Cl = lift_coefficient(omega_vec, v_rel, ball.radius)
    if v < 1e-9:
        Fm = np.zeros(3)
    else:
        lift_dir = np.cross(omega_vec, v_rel)
        n = np.linalg.norm(lift_dir)
        if n > 1e-12:
            lift_dir = lift_dir / n
        Fm = 0.5 * ball.rho_air * Cl * ball.A * (v**2) * lift_dir
    Fg = np.array([0.0, 0.0, -ball.mass * env.g])
    return Fd + Fm + Fg

def rk4_step(state: np.ndarray, dt: float, ball: BallParams, omega_vec: np.ndarray, env) -> np.ndarray:
    def deriv(s):
        v = s[3:]
        a = forces(ball, v, omega_vec, env) / ball.mass
        return np.concatenate([v, a], axis=0)
    k1 = deriv(state)
    k2 = deriv(state + 0.5*dt*k1)
    k3 = deriv(state + 0.5*dt*k2)
    k4 = deriv(state + dt*k3)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def crossed_goal_plane(prev_x, x, goal_x):
    return (prev_x - goal_x) * (x - goal_x) <= 0.0

def check_goal_crossing(prev_state, curr_state, ball: BallParams, env):
    x0, y0, z0 = prev_state[:3]
    x1, y1, z1 = curr_state[:3]
    if x1 == x0:
        alpha = 0.0
    else:
        alpha = (env.goal_x - x0) / (x1 - x0)
    alpha = np.clip(alpha, 0.0, 1.0)
    y_at = y0 + alpha * (y1 - y0)
    z_at = z0 + alpha * (z1 - z0)
    half_w = env.goal_width / 2.0 - ball.radius
    z_min = env.ground_z + ball.radius
    z_max = env.goal_height - ball.radius
    inside = (abs(y_at) <= half_w) and (z_at >= z_min) and (z_at <= z_max)
    point = np.array([env.goal_x, y_at, z_at])
    return inside, point

# def simulate_free_kick_with_goal(kick: KickParams, ball: BallParams = BallParams(), env: EnvParams = EnvParams(),
#                                  x0: float = 0.0, y0: float = 0.0, z0: float = 0.23,
#                                  dt: float = 0.002, t_max: float = 8.0):
#     v0 = initial_velocity(kick.speed, kick.azimuth_deg, kick.elevation_deg)
#     omega_mag = rpm_to_rad_per_s(kick.spin_rpm)
#     spin_axis = angle_to_unit(kick.spin_axis_azimuth_deg, kick.spin_axis_elevation_deg)
#     omega_vec = omega_mag * spin_axis
#     state = np.array([x0, y0, z0, *v0], dtype=float)
#     traj = [state.copy()]
#     times = [0.0]
#     goal_result = {"scored": False, "time": None, "point": None}
#     t = 0.0
#     while t < t_max:
#         prev_state = state.copy()
#         state = rk4_step(state, dt, ball, omega_vec, env)
#         t += dt
#         if state[2] <= env.ground_z and state[5] < 0:
#             p = traj[-1]
#             alpha = (p[2] - env.ground_z) / (p[2] - state[2] + 1e-12)
#             contact = p + alpha * (state - p)
#             contact[2] = env.ground_z
#             traj.append(contact)
#             break
#         if crossed_goal_plane(prev_state[0], state[0], env.goal_x):
#             inside, pt = check_goal_crossing(prev_state, state, ball, env)
#             if inside:
#                 goal_result["scored"] = True
#                 goal_result["time"] = t
#                 goal_result["point"] = pt
#                 traj.append(np.array([pt[0], pt[1], pt[2], *state[3:]]))
#                 break
#         traj.append(state.copy())
#     return np.array(times), np.array(traj), goal_result


def simulate_free_kick_with_goal(
    kick: KickParams,
    ball: BallParams = BallParams(),
    env: EnvParams = EnvParams(),
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.23,
    dt: float = 0.002,
    t_max: float = 8.0,
    sample_interval: float = 0.5,
):
    # Initial conditions
    v0 = initial_velocity(kick.speed, kick.azimuth_deg, kick.elevation_deg)
    omega_mag = rpm_to_rad_per_s(kick.spin_rpm)
    spin_axis = angle_to_unit(kick.spin_axis_azimuth_deg, kick.spin_axis_elevation_deg)
    omega_vec = omega_mag * spin_axis

    state = np.array([x0, y0, z0, *v0], dtype=float)

    # We will store only sampled times / states
    times = [0.0]
    traj = [state.copy()]  # [x, y, z, vx, vy, vz] at t = 0

    goal_result = {"scored": False, "time": None, "point": None}

    t = 0.0
    next_sample_t = sample_interval  # next time at which we want to save a state

    while t < t_max:
        prev_state = state.copy()
        prev_t = t

        # Advance physics by small dt
        state = rk4_step(state, dt, ball, omega_vec, env)
        t += dt

        # --- SAVE STATES AT 0.5s, 1.0s, 1.5s, ... USING INTERPOLATION ---
        # If we've crossed one or more sample times in this step,
        # interpolate between prev_state (at prev_t) and state (at t).
        while next_sample_t <= t:
            alpha = (next_sample_t - prev_t) / (t - prev_t) if t > prev_t else 0.0
            sample_state = prev_state + alpha * (state - prev_state)
            times.append(next_sample_t)
            traj.append(sample_state)
            next_sample_t += sample_interval

        # --- Check ground contact (we don't have to save extra state here) ---
        if state[2] <= env.ground_z and state[5] < 0:
            # Ball has hit the ground; stop simulation.
            break

        # --- Check if we crossed the goal plane and whether it's a goal ---
        if crossed_goal_plane(prev_state[0], state[0], env.goal_x):
            inside, pt = check_goal_crossing(prev_state, state, ball, env)
            if inside:
                goal_result["scored"] = True
                goal_result["time"] = t  # approximate time of crossing
                goal_result["point"] = pt
                break

    return np.array(times), np.array(traj), goal_result
