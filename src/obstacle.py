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

import random
import numpy as np


class DynamicObstacle:
    """
    Class representing a dynamic obstacle that moves through the grid map.
    Each obstacle has a unique ID, a location group ('upper' or 'lower'),
    and moves through defined rows and columns in discrete time steps.
    """
    # Static variable to assign incremental IDs to each obstacle
    _next_id = 1
    # Reference to the shared GridMap instance (set on initialization)
    grid_map = None

    def __init__(self, location, grid_map, move_interval=1):
        """
        Initialize a DynamicObstacle instance.

        Args:
            location (str): Identifier for obstacle group; 'upper' or 'lower'.
            grid_map (GridMap): Reference to the grid map in which the obstacle moves.
            move_interval (int): Number of simulation steps between moves.
                                 Can be customized per obstacle.
        """
        # Store the provided grid map reference as a class variable
        DynamicObstacle.grid_map = grid_map
        self.location = location

        # Internal time state for this obstacle
        self.current_time = 0
        # Whether the obstacle is still active
        self.active = True
        # List of grid cells currently occupied by the obstacle
        self.cells = []

        # Assign a unique ID and increment the class counter
        self.obstacle_id = DynamicObstacle._next_id
        DynamicObstacle._next_id += 1

        # Movement control flags and counters
        self.is_moving = True           # True while obstacle is in motion
        self.move_interval = move_interval  # Steps between moves when in horizontal phase
        self.last_move_time = 0         # Time of last movement

        # Movement phases: vertical approach, horizontal traverse, then exit
        self.phase = "vertical"

        # Set up target rows and starting offsets based on 'upper' or 'lower'
        if location == 'upper':
            # Rows to traverse (6 rows)
            self.target_rows = [15, 16, 17, 18, 19, 20]
            # Fixed column for entrance
            self.current_column = 22
            # Start above the map (negative offset)
            self.current_row_offset = -15
            # Movement direction during vertical phase
            self.move_direction = 1
        else:
            # For 'lower' group: rows below
            self.target_rows = [35, 36, 37, 38, 39, 40]
            self.current_column = 22
            # Start below the map
            self.current_row_offset = 15
            self.move_direction = -1

        # Fixed exit column to traverse towards
        self.exit_column = 61

        # Compute initial cells and place obstacle on the grid
        self.update_cells()
        self.place_on_grid()

        # Force first update to register cells
        self.update()

    def update_cells(self):
        """
        Recompute the list of grid cells occupied by this obstacle
        based on its current row offset and column.
        Removes the obstacle from previous cells before updating.
        """
        # Remove previous markers
        self.remove_from_grid()
        self.cells = []

        # For each target base row, compute actual row position + offset
        for base_row in self.target_rows:
            actual_row = int(base_row + self.current_row_offset)
            # Only include cells that fall inside the map boundaries
            if 0 <= actual_row < DynamicObstacle.grid_map.height:
                # Obstacle spans 3 columns (current_column +/-1)
                for dx in [-1, 0, 1]:
                    cell_x = self.current_column + dx
                    cell_y = actual_row
                    if 0 <= cell_x < DynamicObstacle.grid_map.width:
                        self.cells.append((cell_x, cell_y))

    def place_on_grid(self):
        """
        Mark all current obstacle cells on the real_grid with code -1,
        and record them in grid_map.obstacle_cells dictionary.
        """
        for x, y in self.cells:
            # Only place if coordinates are within valid map range
            if 0 <= x < DynamicObstacle.grid_map.width and 0 <= y < DynamicObstacle.grid_map.height:
                # Mark cell as dynamic obstacle
                DynamicObstacle.grid_map.real_grid[y, x] = -1
                # Record obstacle ID for each cell
                DynamicObstacle.grid_map.obstacle_cells[(x, y)] = self.obstacle_id

    def remove_from_grid(self):
        """
        Clear obstacle markers from the previously occupied cells,
        restoring them to empty space and removing from obstacle_cells map.
        """
        for x, y in self.cells:
            if 0 <= x < DynamicObstacle.grid_map.width and 0 <= y < DynamicObstacle.grid_map.height:
                # Reset cell to empty space code 0
                DynamicObstacle.grid_map.real_grid[y, x] = 0
                # Remove entry from obstacle_cells if present
                DynamicObstacle.grid_map.obstacle_cells.pop((x, y), None)

    def update(self):
        """
        Advance the obstacle's state by one time step, moving it if needed.

        Returns:
            bool: True if obstacle remains active after update, False if it exited and deactivated.
        """
        if not self.active:
            return False

        # Compute time since last move
        elapsed = self.current_time - self.last_move_time
        # Determine interval based on current phase
        interval = self.move_interval if self.phase == "horizontal" else 1
        # Calculate how many move steps to apply
        steps = int(elapsed // interval)
        if steps > 0:
            for _ in range(steps):
                if self.phase == "vertical":
                    # Move vertically toward target rows
                    if self.location == 'upper':
                        if self.current_row_offset < 0:
                            self.current_row_offset += 1
                        else:
                            self.phase = "horizontal"
                    else:
                        if self.current_row_offset > 0:
                            self.current_row_offset -= 1
                        else:
                            self.phase = "horizontal"

                elif self.phase == "horizontal":
                    # Move horizontally toward exit column
                    if self.current_column < self.exit_column:
                        self.current_column += 1
                    elif self.current_column > self.exit_column:
                        self.current_column -= 1
                    # Once at exit column, switch to exit phase
                    if self.current_column == self.exit_column:
                        self.phase = "exit"

                elif self.phase == "exit":
                    # Exit vertically off-screen
                    if self.location == 'upper':
                        if self.current_row_offset > -15:
                            self.current_row_offset -= 1
                        else:
                            # Obstacle has exited the map
                            self.active = False
                            self.remove_from_grid()
                            return False
                    else:
                        if self.current_row_offset < 15:
                            self.current_row_offset += 1
                        else:
                            self.active = False
                            self.remove_from_grid()
                            return False

            # Advance the last_move_time by number of steps processed
            self.last_move_time += steps * interval
            # Update occupied cells and re-place on grid
            self.update_cells()
            self.place_on_grid()

        # Increment internal time counter
        self.current_time += 1
        return True


class ObstacleManager:
    """
    Manager class to create and update multiple DynamicObstacle instances
    using a Poisson process for spawn timing.
    """
    def __init__(self, grid_map):
        """
        Initialize ObstacleManager with spawn rates and initial obstacles.

        Args:
            grid_map (GridMap): reference to the GridMap instance
        """
        self.grid_map = grid_map
        # List of active obstacles
        self.obstacles = []

        # Spawn rates (lambda) per second for upper and lower obstacles
        self.lambda_upper = 1/1500.0
        self.lambda_lower = 1/1500.0

        # Immediately create one obstacle for each location group
        self._create_initial_obstacles()

    def _create_initial_obstacles(self):
        """
        Generate initial obstacles at simulation start for both groups.
        """
        # Create an initial upper obstacle with slower movement
        upper = DynamicObstacle('upper', self.grid_map, move_interval=10)
        self.obstacles.append(upper)
        # Create an initial lower obstacle with faster movement
        lower = DynamicObstacle('lower', self.grid_map, move_interval=4)
        self.obstacles.append(lower)
        print("Initial obstacles created.")

    def update(self, current_time):
        """
        Update all existing obstacles, remove inactive ones,
        and potentially spawn new obstacles based on Poisson rates.

        Args:
            current_time (float): current simulation time

        Returns:
            bool: True if any obstacles were added or removed
        """
        active_list = []
        changed = False

        # Update each obstacle; keep if still active
        for obs in self.obstacles:
            if obs.update():
                active_list.append(obs)
        self.obstacles = active_list

        # Swap lambdas at a specific time threshold
        if current_time == 15000:
            self.lambda_lower, self.lambda_upper = self.lambda_upper, self.lambda_lower
            print(f"At time {current_time}, swapped spawn rates: "
                  f"lambda_lower={self.lambda_lower}, lambda_upper={self.lambda_upper}")

        # Spawn new upper obstacles if none active
        if not any(o.location=='upper' and o.active for o in self.obstacles):
            count = np.random.poisson(self.lambda_upper)
            for _ in range(count):
                interval = 10 if current_time < 15000 else 3
                new_obs = DynamicObstacle('upper', self.grid_map, move_interval=interval)
                self.obstacles.append(new_obs)
                changed = True

        # Spawn new lower obstacles if none active
        if not any(o.location=='lower' and o.active for o in self.obstacles):
            count = np.random.poisson(self.lambda_lower)
            for _ in range(count):
                interval = 3 if current_time < 15000 else 10
                new_obs = DynamicObstacle('lower', self.grid_map, move_interval=interval)
                self.obstacles.append(new_obs)
                changed = True

        return changed
