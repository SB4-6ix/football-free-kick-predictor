# ============================================================
# 1. Import Libraries
# ============================================================

from free_kick_simulator import BallParams, KickParams, EnvParams, simulate_free_kick_with_goal
import pandas as pd

# ============================================================
# 2. Set Simulation Conditions
# ============================================================

ball = BallParams(rho_air=1.20, Cd=0.28)                    # set air density/drag
env  = EnvParams(wind_xyz=(2.0, -1.0, 0.0), goal_x=22.0)    # wind + goal geometry

# ============================================================
# 3. Simulation Parameters
# ============================================================
# Initial ball speed: 25 to 32 m/s (increments of 0.5)
speed_params = [25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32]
# Launch angle: 14 to 19 degrees (increments of 0.2)
elevation_params = [14, 14.2, 14.4, 14.6, 14.8, 15, 15.2, 15.4, 15.6, 15.8, 16, 16.2, 16.4, 16.6, 16.8, 17, 17.2, 17.4, 17.6, 17.8, 18, 18.2, 18.4, 18.6, 18.8, 19] #26
# Ball spin: 0 to 350 RPM (increments of 25)
rpm_params = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350] #14

# ============================================================
# 4. Generate Free-Kick Simulation Data
# ============================================================

# Initialize storage and simulation counter
all_results_list = []
simulation_count = 0

# Run simulations for all combinations of speed, elevation, and spin
for speed in speed_params:  
    for elevation_deg in elevation_params: 
        for spin_rpm in rpm_params:  
            
            # Define kick conditions
            kick = KickParams(
                speed=speed,
                azimuth_deg=0.5,
                elevation_deg=elevation_deg,
                spin_rpm=spin_rpm,
                spin_axis_azimuth_deg=60.0,
                spin_axis_elevation_deg=8
            )
            
            # Run the simulation
            t, traj, result = simulate_free_kick_with_goal(kick, ball, env, z0=0.23)

            # Store simulation inputs and outcomes
            row_data = {
                'speed': speed,
                'elevation degree': elevation_deg,
                'spin rpm': spin_rpm,

                "result_time": result["time"],     # Extracted the time of flight
                'result_scored': result["scored"], # Extracted the 'scored' boolean
                'result_point': result["point"]    # Extracted the 'point' data
            }
            all_results_list.append(row_data)

            # Ensure these lines have the EXACT same leading spaces/tabs as the line above
            simulation_count += 1
            if simulation_count % 1000 == 0:
                print(f"Status: Completed {simulation_count} simulations.")

# ============================================================
# 5. Export Dataset to CSV
# ============================================================

print(f"Simulation complete. Total runs recorded: {len(all_results_list)}")

# Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(all_results_list)

# Save the DataFrame to a CSV file
from pathlib import Path
folder = Path(__file__).resolve().parent
df.to_csv(folder / "data.csv", index=False)

print("Successfully saved all data to data.csv")