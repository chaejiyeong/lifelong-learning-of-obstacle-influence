"""
------------------------------------------------------------------------------
Copyright (c) 2025 Anonymous Authors. All rights reserved.

This source code accompanies the manuscript entitled
"Cooperative Probabilistic Costmap with Lifelong Learning for Multi‑Robot Navigation,"
submitted to the Conference on Robot Learning (CoRL) 2025.

Unauthorized reproduction, distribution, or modification of this code in any form
without the express written consent of the authors is strictly prohibited.
------------------------------------------------------------------------------
"""
import time
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QDoubleSpinBox, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
import csv

from grid_map import GridMap
from robot import Robot
from obstacle import DynamicObstacle
import numpy as np
import math
from matplotlib.patches import Circle


def f(t, lam):
    """
    Compute the obstacle influence function f(t) = exp(-lam * (ln(t))^2) + k.
    Returns 0 at t=0 to avoid log(0) singularity.

    Args:
        t (float): elapsed time since last observation
        lam (float): decay parameter for obstacle influence

    Returns:
        float: influence value, or 0 if t == 0
    """
    if t == 0:
        return 0
    return math.exp(-lam * (math.log(t) ** 2))


class GridMapWidget(QWidget):
    """
    PyQt5 widget for simulating and visualizing a grid map,
    dynamic obstacles, and robot missions.
    """
    def __init__(self, parent=None, file_suffix=""):
        """
        Initialize the widget, simulation parameters, and UI.

        Args:
            parent (QWidget, optional): parent widget reference
            file_suffix (str, optional): suffix for log filenames
        """
        super().__init__(parent)
        self.file_suffix = file_suffix

        # Simulation timing parameters
        self.simulation_time = 0.0
        self.time_scale = 0.5
        self.simulation_duration = 30000.0  # total simulation duration in seconds

        # Grid map model with specified dimensions
        self.grid_map = GridMap(83, 55)

        # Robot management lists and counters
        self.active_robots = []              # robots currently active in simulation
        self.inactive_robots = []            # robots completed routes, awaiting reuse
        self.max_robots = 6                  # maximum concurrent robots
        self.next_robot_id = 0               # next ID to assign when spawning

        # Timer to drive simulation updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.simulation_running = False

        # Robot spawning timing details
        self.robot_spawn_interval = 60       # spawn a new robot every N seconds
        self.last_spawn_time_even = 0        # last spawn time for even-ID robots
        self.last_spawn_time_odd = 0         # last spawn time for odd-ID robots

        # Base timer interval at 1x speed (1000 ms per simulated second)
        self.base_simulation_speed = 1000

        # Counters for mission statistics
        self.smart_route_count = 0
        self.dumb_route_count = 0
        self.mission_complete_count = 0
        self.avg_mission_completions_per_hour = 0.0

        # Option to disable GUI updates for headless or faster runs
        self.disable_gui_updates = False

        # Build UI components and layout
        self._init_ui()

        # Draw initial map state
        self.update_plot()

    def _init_ui(self):
        """
        Set up UI panels, plots, controls, and statistics display.
        """
        # Main vertical layout
        self.layout = QVBoxLayout(self)

        # Panel to hold map canvases horizontally
        self.maps_panel = QWidget()
        self.maps_layout = QHBoxLayout(self.maps_panel)

        # Lower panel for controls and stats
        self.lower_panel = QWidget()
        self.lower_layout = QHBoxLayout(self.lower_panel)

        # Actual environment plot (left)
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.maps_layout.addWidget(self.canvas)

        # Shared perception plot (right)
        self.shared_figure, self.shared_ax = plt.subplots(figsize=(12, 8))
        self.shared_canvas = FigureCanvas(self.shared_figure)
        self.maps_layout.addWidget(self.shared_canvas)

        # Group box for simulation controls
        self.control_group = QGroupBox("Simulation Controls")
        self.control_layout = QGridLayout(self.control_group)

        # Start/stop simulation button
        self.start_stop_button = QPushButton("Start Simulation")
        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.control_layout.addWidget(self.start_stop_button, 0, 0, 1, 2)

        # Label and spinbox for time scaling
        self.time_scale_label = QLabel("Time Scale:")
        self.control_layout.addWidget(self.time_scale_label, 1, 0)
        self.time_scale_spinbox = QDoubleSpinBox()
        self.time_scale_spinbox.setRange(1.0, 300.0)
        self.time_scale_spinbox.setSingleStep(1.0)
        self.time_scale_spinbox.setValue(1.0)
        self.time_scale_spinbox.setSuffix("x")
        self.time_scale_spinbox.valueChanged.connect(self.update_time_scale)
        self.control_layout.addWidget(self.time_scale_spinbox, 1, 1)

        # Group box for statistics
        self.stats_group = QGroupBox("Statistics and Information")
        self.stats_layout = QVBoxLayout(self.stats_group)

        # Route definition info
        self.route_info = QLabel("Routes: A→B→C→A (even IDs), D→E→F→D (odd IDs)")
        self.stats_layout.addWidget(self.route_info)

        # Display current simulation time and speed
        self.time_info = QLabel("Simulation Time: 0.0s")
        self.stats_layout.addWidget(self.time_info)

        # Mission statistics labels
        self.mission_complete_label = QLabel("Total Missions Completed: 0")
        self.smart_route_label     = QLabel("Smart Route Count: 0")
        self.dumb_route_label      = QLabel("Dumb Route Count: 0")
        self.avg_mission_label     = QLabel("Avg Missions per Hour: 0.0")
        self.stats_layout.addWidget(self.mission_complete_label)
        self.stats_layout.addWidget(self.smart_route_label)
        self.stats_layout.addWidget(self.dumb_route_label)
        self.stats_layout.addWidget(self.avg_mission_label)

        # Button to refresh mission statistics popup
        self.update_mission_stats_button = QPushButton("Show Mission Statistics")
        self.update_mission_stats_button.clicked.connect(self.update_mission_stats)
        self.stats_layout.addWidget(self.update_mission_stats_button)

        # Button to manually save mission logs to CSV
        self.save_stats_button = QPushButton("Save Mission Logs")
        self.save_stats_button.clicked.connect(self.save_mission_logs)
        self.stats_layout.addWidget(self.save_stats_button)

        # Button to load and apply trained lambda values
        self.apply_lambda_button = QPushButton("Apply Trained Lambda Values")
        self.apply_lambda_button.clicked.connect(self.apply_learned_lambda_values)
        self.stats_layout.addWidget(self.apply_lambda_button)

        # Combine control and stats groups vertically
        self.control_stats_group = QGroupBox("Control & Stats Panel")
        self.control_stats_layout = QVBoxLayout(self.control_stats_group)
        self.control_stats_layout.addWidget(self.control_group)
        self.control_stats_layout.addWidget(self.stats_group)

        # Add static FT graphs group
        self.static_info_group = QGroupBox("Static Influence Curves")
        self.static_info_layout = QHBoxLayout(self.static_info_group)
        self.static_positions = [(22,10), (22,20), (22,40), (22,50)]
        self.last_ft_update = 0.0
        self.update_ft_graphs()

        # Assemble lower panel: static graphs on left, controls on right
        self.lower_layout.addWidget(self.static_info_group, 7)
        self.lower_layout.addWidget(self.control_stats_group, 3)

        # Add panels to main layout
        self.layout.addWidget(self.maps_panel, 9)
        self.layout.addWidget(self.lower_panel, 1)

    def get_simulation_time(self):
        """
        Return the current simulation time.
        """
        return self.simulation_time

    def update_time_scale(self, value):
        """
        Update the simulation speed multiplier.

        Args:
            value (float): new time scale multiplier
        """
        self.time_scale = value

    def update_simulation(self):
        """
        Main simulation loop called by QTimer on each timeout.
        Advances simulation time, updates map, spawns/moves robots, and updates UI.
        """
        # Determine how many simulation steps to advance per timer tick
        steps = int(self.time_scale) if not self.disable_gui_updates else 1000
        for _ in range(steps):
            self.simulation_time += 1
            t = self.simulation_time

            # Update dynamic obstacles
            self.grid_map.update_obstacles(t)
            # Update inflation/influence layer
            self.grid_map.update_inflation_grid()
            # Handle new robot creation
            self._process_robot_creation(t)
            # Robot sensing, replanning, and movement
            self._process_robot_movement()

            # Print progress every 1000s
            if t % 1000 == 0:
                print(f"Simulating: time={t}s, completed={self.mission_complete_count}, smart={self.smart_route_count}, dumb={self.dumb_route_count}")

            # End simulation if duration reached
            if t >= self.simulation_duration:
                self.timer.stop()
                self.simulation_running = False
                self.start_stop_button.setText("Start Simulation")
                self.time_scale_label.setText("Time Scale: (stopped)")
                # Auto-save logs and lambda values
                self.auto_save_mission_logs(f"logs/simulation_log{self.file_suffix}.csv")
                self.save_lambda_values(f"logs/lambda_values{self.file_suffix}.csv")
                print(f"Simulation finished at t={t}s, missions done={self.mission_complete_count}")
                return

        # Update plots and UI once after stepping
        if not self.disable_gui_updates:
            self._update_ui(self.simulation_time)

    def _process_robot_creation(self, current_time):
        """
        Spawn new robots at configured intervals, balancing even/odd IDs.
        """
        desired_per_group = self.max_robots // 2
        count_even = sum(1 for r in self.active_robots if r.robot_id % 2 == 0)
        count_odd  = sum(1 for r in self.active_robots if r.robot_id % 2 == 1)

        # Spawn for even IDs
        if count_even < desired_per_group and current_time - self.last_spawn_time_even >= self.robot_spawn_interval:
            self.spawn_robot(parity=0)
            self.last_spawn_time_even = current_time

        # Spawn for odd IDs
        if count_odd < desired_per_group and current_time - self.last_spawn_time_odd >= self.robot_spawn_interval:
            self.spawn_robot(parity=1)
            self.last_spawn_time_odd = current_time

    def _process_robot_movement(self):
        """
        Update each robot: sense environment, replan path, move, and collect statistics.
        """
        next_active = []

        # Sensor update
        for robot in self.active_robots:
            robot.perceive_environment()
        # Influence grid update
        self.grid_map.update_ft_grid()
        # Replan paths
        for robot in self.active_robots:
            robot.plan_next_path()
        # Move robots
        for robot in self.active_robots:
            robot.move()

        # Check completion
        for robot in self.active_robots:
            if robot.route_completed:
                # Log mission end
                self.mission_complete_count += 1
                Robot.mission_log.append((self.simulation_time, robot.route_type, robot.journey_time))
                # Update counters based on route correctness and time
                if robot.paths_different:
                    self.dumb_route_count += 1
                else:
                    self.smart_route_count += 1
                # Add robot back to inactive pool
                self.inactive_robots.append(robot)
            else:
                next_active.append(robot)

        self.active_robots = next_active

    def _update_ui(self, current_time):
        """
        Refresh plots and labels in the GUI based on current simulation state.
        """
        # Redraw map and robots
        self.update_plot()
        # Update statistics labels
        self.time_info.setText(f"Simulation Time: {current_time:.1f}s (x{self.time_scale:.1f})")
        self.mission_complete_label.setText(f"Total Missions Completed: {self.mission_complete_count}")
        self.smart_route_label.setText(f"Smart Route Count: {self.smart_route_count}")
        self.dumb_route_label.setText(f"Dumb Route Count: {self.dumb_route_count}")
        self.avg_mission_label.setText(f"Avg Missions per Hour: {self.avg_mission_completions_per_hour:.1f}")
        # Periodically update influence curves
        if current_time - self.last_ft_update >= 100:
            self.update_ft_graphs()
            self.last_ft_update = current_time

    def reset_robot(self, robot):
        """
        Reset a robot's state for reuse: clear path, reset timers, and replanning.
        """
        robot.paths_different = False
        robot.route_completed = False
        robot.journey_time = 0
        robot.current_target_idx = 0
        robot.path = []
        robot.waiting = False
        robot.stuck_time = 0
        robot.plan_next_path()

    def spawn_robot(self, parity=None):
        """
        Create a new Robot instance at the start of its route.

        Args:
            parity (int, optional): 0 for even-ID route, 1 for odd-ID route
        """
        # Determine next ID
        self.next_robot_id = (self.next_robot_id % self.max_robots) + 1
        robot_id = self.next_robot_id
        route_type = parity if parity is not None else robot_id % 2

        # Find start point by route type
        start_pt = next((p for p in self.grid_map.mission_points if (route_type == 0 and p[2]=='A') or (route_type==1 and p[2]=='D')), None)
        if not start_pt:
            return None
        robot = Robot(start_pt, self.grid_map.mission_points, self.grid_map, self.simulation_time, route_type, robot_id)
        self.active_robots.append(robot)
        return robot_id

    def update_plot(self):
        """
        Draw the actual and shared grid maps, including dynamic obstacles and robots.
        """
        # Clear and plot actual map
        self.ax.clear()
        self.grid_map.plot(self.ax, self.active_robots)
        # Draw robot markers
        for robot in self.active_robots:
            x, y = robot.get_position()
            color = '#ff4040' if robot.paths_different else '#4080ff'
            outer = Circle((x,y), 0.65, color=color, alpha=0.8)
            inner = Circle((x,y), 0.45, color=color, alpha=0.9)
            self.ax.add_patch(outer)
            self.ax.add_patch(inner)
            self.ax.text(x, y, f"{robot.robot_id}", ha='center', va='center', color='white', fontsize=8)
        self.canvas.draw()

        # Clear and plot shared perception map
        self.shared_ax.clear()
        self.grid_map.plot_shared_perception(self.shared_ax)
        self.shared_canvas.draw()

    def toggle_simulation(self):
        """
        Toggle simulation on/off and adjust UI accordingly.
        """
        if self.simulation_running:
            self.timer.stop()
            self.simulation_running = False
            self.start_stop_button.setText("Start Simulation")
            self.time_scale_label.setText("Time Scale: (stopped)")
        else:
            self.simulation_time = 0.0
            self.time_scale = 1.0
            self.time_scale_label.setText("Time Scale: (running)")
            interval = max(10, min(100, int(self.base_simulation_speed / self.time_scale)))
            self.timer.start(interval)
            self.simulation_running = True
            self.start_stop_button.setText("Stop Simulation")

    def save_mission_logs(self):
        """
        Prompt for filename and save mission logs and summary stats to a CSV.
        """
        filename, ok = QInputDialog.getText(self, "Save Logs", "Enter CSV filename:")
        if ok:
            if not filename.endswith('.csv'):
                filename += '.csv'
            try:
                with open(f"logs/{filename}", 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Time", "RouteType", "JourneyTime"]);
                    for rec in Robot.mission_log:
                        writer.writerow(rec)
                    writer.writerow([])
                    writer.writerow(["Completed", "Smart", "Dumb"]);
                    writer.writerow([self.mission_complete_count, self.smart_route_count, self.dumb_route_count])
                QMessageBox.information(self, "Saved", f"Logs saved to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save logs: {e}")

    def auto_save_mission_logs(self, filename):
        """
        Automatically save mission logs without user prompt.
        """
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time", "RouteType", "JourneyTime"]);
                for rec in Robot.mission_log:
                    writer.writerow(rec)
                writer.writerow([])
                writer.writerow(["Completed", "Smart", "Dumb"]);
                writer.writerow([self.mission_complete_count, self.smart_route_count, self.dumb_route_count])
            print(f"Auto-saved logs to {filename}")
        except Exception as e:
            print(f"Error auto-saving logs: {e}")

    def update_mission_stats(self):
        """
        Show a message box with detailed mission statistics.
        """
        text = "Mission Statistics:\n"
        for (route, time), count in Robot.mission_stats.items():
            text += f"Route {route}, Time {time:.1f}s: {count}\n"
        QMessageBox.information(self, "Mission Stats", text)

    def update_ft_graphs(self):
        """
        Refresh static influence function plots for selected cells.
        """
        # Clear existing plots
        while self.static_info_layout.count():
            item = self.static_info_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Generate small plots for each fixed position
        for (x,y) in self.static_positions:
            lam = self.grid_map.shared_ft_grid[y, x]['lambda']
            fig, ax = plt.subplots(figsize=(3,2))
            tvals = np.linspace(0.1, 100, 100)
            fvals = [math.exp(-lam*(math.log(t)**2)) for t in tvals]
            ax.plot(tvals, fvals)
            ax.set_title(f"Cell ({x},{y}) λ={lam:.2f}")
            ax.set_xlabel("t")
            ax.set_ylabel("f(t)")
            canvas = FigureCanvas(fig)
            self.static_info_layout.addWidget(canvas)

    def save_lambda_values(self, filename):
        """
        Save all grid cell lambda values to CSV.
        """
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["X", "Y", "Lambda"]);
                for y in range(self.grid_map.height):
                    for x in range(self.grid_map.width):
                        lam = self.grid_map.shared_ft_grid[y, x]['lambda']
                        writer.writerow([x, y, f"{lam:.2f}"])
            print(f"Saved lambda values to {filename}")
        except Exception as e:
            print(f"Error saving lambda values: {e}")

    def apply_learned_lambda_values(self):
        """
        Load trained lambda values from CSV and apply to shared grid.
        """
        try:
            with open(f"logs/lambda_values{self.file_suffix}.csv", 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # skip header
                for row in reader:
                    x, y, lam = int(row[0]), int(row[1]), float(row[2])
                    if 0 <= x < self.grid_map.width and 0 <= y < self.grid_map.height:
                        self.grid_map.shared_ft_grid[y, x]['lambda'] = lam
            QMessageBox.information(self, "Done", "Applied trained lambda values.")
            self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply lambda values: {e}")

if __name__ == '__main__':
    import sys
    import argparse
    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser(description='GridMapWidget Simulator')
    parser.add_argument('suffix', nargs='?', default='', help='Filename suffix for logs')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    widget = GridMapWidget(file_suffix=args.suffix)
    widget.show()
    sys.exit(app.exec_())
