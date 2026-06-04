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
import heapq
import numpy as np
import math


def f(t, lam, k=0):
    """
    Compute the obstacle influence function f(t) = exp(-lam * (ln(t))^2) + k.
    Returns 0 at t=0 to avoid log(0) errors.

    Args:
        t (float): elapsed time since obstacle detection
        lam (float): decay rate parameter
        k (float, optional): constant offset term (default=0)

    Returns:
        float: influence value, or 0 if t == 0
    """
    if t == 0:
        return 0
    return math.exp(-lam * (math.log(t) ** 2)) + k


def df(t, lam):
    """
    Compute the derivative of the influence function df/dt.
    Returns 0 at t=0 to avoid log(0) errors.

    Args:
        t (float): elapsed time
        lam (float): decay rate parameter

    Returns:
        float: derivative value, or 0 if t == 0
    """
    if t == 0:
        return 0
    return -2 * lam * math.log(t) * math.exp(-lam * (math.log(t) ** 2))


def cal_k(t, lam, N):
    """
    Compute offset parameter k based on time t, decay lam, and history depth N.

    Args:
        t (float): elapsed time
        lam (float): decay rate parameter
        N (int): history depth parameter

    Returns:
        float: computed offset k
    """
    return 0.5 - math.exp(-lam * (math.log(t + N) ** 2))


def get_cost_color(fval):
    """
    Determine overlay color based on influence value fval.

    Color mapping:
      - fval > 0.5: pure red (#FF0000)
      - 0.1 < fval <= 0.5: linear interpolation from blue (#0000FF) at 0.1 to red at 0.5
      - fval <= 0.1: white (#FFFFFF)

    Args:
        fval (float): influence value

    Returns:
        str: hex color code
    """
    try:
        fval = float(fval)
    except Exception:
        return "#FFFFFF"

    if fval > 0.5:
        return "#FF0000"
    elif fval > 0.1:
        # Normalize between 0.1 and 0.5
        t = (fval - 0.1) / 0.4  # t=0 at fval=0.1, t=1 at fval=0.5
        # Interpolate red and blue channels
        R = int(t * 255)
        G = 0
        B = int((1 - t) * 255)
        return '#{:02X}{:02X}{:02X}'.format(R, G, B)
    else:
        return "#FFFFFF"


class Robot:
    """
    Robot class representing an autonomous agent navigating the grid map.

    Robots plan paths through mission points, detect and avoid obstacles,
    and log performance statistics.
    """
    # Class variables for aggregated mission statistics
    dumb_route_count = 0
    smart_route_count = 0
    mission_complete_count = 0
    # Shared grid map reference, set at robot creation
    grid_map = None
    # Dictionary mapping (route_type, journey_time) to count of completions
    mission_stats = {}
    # List of detailed mission log entries (time, route_type, journey_time)
    mission_log = []

    def __init__(self, start_pos, mission_points, grid_map, current_time,
                 route_type=None, robot_id=None):
        """
        Initialize a Robot instance.

        Args:
            start_pos (tuple): starting coordinates (x, y)
            mission_points (list): list of tuples (x, y, label) for mission targets
            grid_map (GridMap): reference to shared grid map instance
            current_time (float): simulation start time
            route_type (int, optional): 0 for A->B->C->A, 1 for D->E->F->D; random if None
            robot_id (int, optional): unique identifier; random if None
        """
        # Current position
        self.pos = (start_pos[0], start_pos[1])
        # List of mission points (x, y, type)
        self.mission_points = mission_points
        # Store grid map reference for path planning and obstacle queries
        Robot.grid_map = grid_map

        # State flags and counters
        self.route_completed = False
        self.waiting = False       # True if robot is blocked
        self.waiting_time = 0      # consecutive wait steps
        self.stuck_time = 0        # time spent in same cell
        self.journey_time = 0      # travel time excluding waits
        self.start_time = current_time
        # Last movement direction vector for subsequent perception
        self.last_move_dir = (0, 0)
        # Unique robot identifier
        self.robot_id = robot_id if robot_id is not None else random.randint(1000, 9999)

        # Route type: 0 or 1
        self.route_type = route_type if route_type is not None else random.randint(0, 1)
        # Flags for path analysis
        self.detour_route = False
        self.correct_route = False
        self.paths_different = False
        self.obstacle_detected = False

        # Build index mapping for mission point labels
        self.point_indices = {}
        for idx, (_, _, label) in enumerate(mission_points):
            if label not in self.point_indices:
                self.point_indices[label] = idx

        # Initialize experience grid for f(t) state: IDEAL, OBSTACLE, AVOID, etc.
        height = Robot.grid_map.height
        width = Robot.grid_map.width
        self.exp_grid = np.empty((height, width), dtype=object)
        for i in range(height):
            for j in range(width):
                self.exp_grid[i, j] = {'exp': 'IDEAL'}

        # Determine mission route sequence based on route_type
        if self.route_type == 0 and 'A' in self.point_indices:
            a = self.point_indices['A']
            if 'B' in self.point_indices and 'C' in self.point_indices:
                self.route = [a, self.point_indices['B'], self.point_indices['C'], a]
            else:
                self.route = [0]
        elif self.route_type == 1 and 'D' in self.point_indices:
            d = self.point_indices['D']
            if 'E' in self.point_indices and 'F' in self.point_indices:
                self.route = [d, self.point_indices['E'], self.point_indices['F'], d]
            else:
                self.route = [0]
        else:
            # Fallback to first point only
            self.route = [0]

        # Initialize path planning variables
        self.current_target_idx = 0
        self.path = []
        self.path_index = 0

        # Plan initial path towards first target
        self.plan_next_path()

    def get_position(self):
        """
        Return the robot's current position.

        Returns:
            tuple: (x, y) coordinates
        """
        return self.pos

    def get_current_target(self):
        """
        Return the label of the current mission target.

        Returns:
            str or None: mission label (e.g., 'A', 'B') or None if completed
        """
        if self.current_target_idx < len(self.route):
            idx = self.route[self.current_target_idx]
            return self.mission_points[idx][2]
        return None

    def plan_next_path(self):
        """
        Compute the shortest path from current position to the next target
        using cooperative costmap and inflation map, and compare paths.

        Returns:
            bool: True if path planned successfully, False otherwise
        """
        # Check if all targets completed
        if self.current_target_idx >= len(self.route):
            self.route_completed = True
            return False

        # Determine target coordinates
        target_idx = self.route[self.current_target_idx]
        tx, ty, _ = self.mission_points[target_idx]
        target_pos = (tx, ty)

        # If already at target, advance to next
        if self.pos == target_pos:
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.route):
                self.route_completed = True
                return True
            return self.plan_next_path()

        # Compute path on shared perception grid (with obstacles)
        self.path = self._calculate_path_with_grid(
            self.pos, target_pos, Robot.grid_map.shared_perception_grid
        )
        # Compare against inflation grid path to detect detours
        if not self.paths_different:
            original = self._calculate_path_with_grid(
                self.pos, target_pos, Robot.grid_map.inflation_grid
            )
            self.paths_different = self._compare_paths(self.path, original)

        # If no path found, enter waiting state
        if not self.path:
            self.path = [self.pos]
            self.path_index = 0
            self.waiting = True
            return False

        # Remove current cell if first in path
        if len(self.path) > 1 and self.path[0] == self.pos:
            self.path.pop(0)
        # Re-check empty path
        if not self.path:
            self.current_target_idx += 1
            if self.current_target_idx >= len(self.route):
                self.route_completed = True
                return True
            return self.plan_next_path()

        # Reset movement counters
        self.path_index = 0
        self.stuck_time = 0
        self.waiting = False
        return True

    def move(self):
        """
        Move the robot one step along its planned path if possible.

        Returns:
            bool: True if movement occurred, False otherwise
        """
        # If waiting or no path or completed, do not move
        if self.waiting or not self.path or self.route_completed:
            return False

        # If path index beyond end, stay or replan next
        if self.path_index >= len(self.path):
            return False

        # Determine next candidate position
        candidate = self.path[self.path_index]
        # Check for obstacles or walls in real grid
        x, y = candidate
        grid_val = Robot.grid_map.real_grid[y, x]
        if grid_val in (1, -1):
            # Blocked, enter waiting
            self.waiting = True
            return False

        # Proceed to next position
        prev_pos = self.pos
        self.pos = candidate
        self.path_index += 1

        # Update movement direction
        dx = self.pos[0] - prev_pos[0]
        dy = self.pos[1] - prev_pos[1]
        if dx or dy:
            self.last_move_dir = (int(math.copysign(1, dx)) if dx else 0,
                                  int(math.copysign(1, dy)) if dy else 0)

        # Log journey time for specific segments
        if self.get_current_target() in ['B', 'E']:
            self.journey_time += 1

        # Detect passing key waypoints for correct/ detour flags
        if self.pos in [(24, 18), (24, 37)]:
            self.correct_route = True
        if self.pos in [(57, 26), (57, 29)]:
            self.detour_route = True

        # Update stuck_time or reset
        if self.pos == prev_pos:
            self.stuck_time += 1
        else:
            self.stuck_time = 0

        return True

    def is_waiting(self):
        """
        Check if robot is in waiting (blocked) state.

        Returns:
            bool: True if waiting, False otherwise
        """
        return self.waiting

    def perceive_environment(self):
        """
        Scan the area ahead according to movement direction,
        detect obstacles, and update shared perception.

        Returns:
            bool: True if obstacle detected, False otherwise
        """
        x, y = self.pos
        detected = False
        obstacle_positions = []

        # Determine scan direction: use path or last move direction
        move_dir = self.last_move_dir
        if self.path and self.path_index < len(self.path):
            nx, ny = self.path[self.path_index]
            move_dir = (int(math.copysign(1, nx - x)) if nx!=x else 0,
                        int(math.copysign(1, ny - y)) if ny!=y else 0)

        # Only horizontal or vertical scans
        if move_dir[0] and not move_dir[1]:
            # Scan up to 9 cells ahead horizontally, +/-3 rows
            for d in range(1, 10):
                for offset in range(-3, 4):
                    sx = x + d * move_dir[0]
                    sy = y + offset
                    if self._is_obstacle_at_position(sx, sy):
                        detected = True
                        obstacle_positions.append((sx, sy))
                        self._check_cell(sx, sy)
        elif move_dir[1] and not move_dir[0]:
            # Scan up to 9 cells ahead vertically, +/-3 columns
            for d in range(1, 10):
                for offset in range(-3, 4):
                    sx = x + offset
                    sy = y + d * move_dir[1]
                    if self._is_obstacle_at_position(sx, sy):
                        detected = True
                        obstacle_positions.append((sx, sy))
                        self._check_cell(sx, sy)
        else:
            # Fallback scan 2-cell radius
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx==0 and dy==0: continue
                    sx, sy = x+dx, y+dy
                    if self._is_obstacle_at_position(sx, sy):
                        detected = True
                        obstacle_positions.append((sx, sy))
                        self._check_cell(sx, sy)

        # If obstacles found, update f(t) for certain positions
        if detected and len(obstacle_positions) > 5 and \
           len(obstacle_positions) % 6 == 0 and \
           all(pos[0] == obstacle_positions[0][0] for pos in obstacle_positions):
            for sx, sy in obstacle_positions:
                if move_dir[0] and sy in [6, 18, 37, 49]:
                    self.obstacle_ft_param(sx, sy)

        self.obstacle_detected = detected
        return detected

    def _compare_paths(self, path1, path2):
        """
        Compare two paths for equality.

        Returns True if they differ, False if identical.
        """
        return path1 != path2

    def _calculate_path_with_grid(self, start, end, grid):
        """
        Compute shortest path from start to end using Dijkstra's algorithm
        on the given grid, treating walls and obstacles as impassable.

        Args:
            start (tuple): (x, y) start coordinates
            end (tuple): (x, y) goal coordinates
            grid (ndarray): 2D map array with codes: 0 empty, 1 wall, -1 obstacle,
                             inflation uses high cost

        Returns:
            list: sequence of (x, y) coordinates for the path
        """
        if start == end:
            return [start]
        height, width = grid.shape
        # Movement directions (up, right, down, left)
        directions = [(0,-1), (1,0), (0,1), (-1,0)]
        # Distance map
        dist = np.full((height,width), np.inf)
        dist[start[1], start[0]] = 0
        visited = np.zeros((height,width), dtype=bool)
        # Min-heap priority queue
        pq = [(0, start[0], start[1])]
        prev = {}
        while pq:
            cur_dist, x, y = heapq.heappop(pq)
            if visited[y,x]: continue
            visited[y,x] = True
            if (x,y) == end:
                break
            for dx,dy in directions:
                nx, ny = x+dx, y+dy
                if 0<=nx<width and 0<=ny<height and not visited[ny,nx]:
                    val = grid[ny,nx]
                    if val in (1, -1):
                        continue
                    cost = 10 if val==Robot.grid_map.INFLATION_VALUE else 1
                    nd = cur_dist + cost
                    if nd < dist[ny,nx]:
                        dist[ny,nx] = nd
                        prev[(nx,ny)] = (x,y)
                        heapq.heappush(pq, (nd, nx, ny))
        # Reconstruct path
        path = []
        cur = end
        if cur not in prev and start!=end:
            # find closest reachable node to end
            min_m = np.inf
            closest = start
            for yy in range(height):
                for xx in range(width):
                    if visited[yy,xx]:
                        m = abs(xx-end[0])+abs(yy-end[1])
                        if m<min_m:
                            min_m=m; closest=(xx,yy)
            if closest==start:
                return []
            cur = closest
        while cur!=start:
            path.append(cur)
            cur = prev.get(cur, start)
        path.append(start)
        path.reverse()
        return path

    def _is_obstacle_at_position(self, x, y):
        """
        Check if a position contains a dynamic obstacle that was not in the base map.

        Returns True if real_grid has -1 at (x,y) and base_grid was empty.
        """
        if not (0<=x<Robot.grid_map.width and 0<=y<Robot.grid_map.height):
            return False
        base = Robot.grid_map.base_grid[y,x]
        real = Robot.grid_map.real_grid[y,x]
        return base==0 and real==-1

    def _check_cell(self, x, y):
        """
        Update shared perception map if an obstacle is detected at cell.

        Returns True if shared perception updated, False otherwise.
        """
        if not (0<=x<Robot.grid_map.width and 0<=y<Robot.grid_map.height):
            return False
        # If already marked, skip
        if Robot.grid_map.shared_perception_grid[y,x]==-1:
            return False
        # If real obstacle present, update perception
        if Robot.grid_map.real_grid[y,x]==-1:
            Robot.grid_map.update_shared_perception(x,y)
            return True
        return False

    def obstacle_ft_param(self, x, y):
        """
        Update experience grid f(t) state when encountering obstacles.
        Transitions: IDEAL->OBSTACLE->AVOID on repeated encounters.
        """
        cell = self.exp_grid[y,x]
        explored = Robot.grid_map.shared_ft_grid[y,x]['explored']
        if not explored:
            # First encounter
            if cell['exp']=='IDEAL':
                cell['exp']='OBSTACLE'
            elif cell['exp']=='OBSTACLE':
                cell['exp']='AVOID'
        else:
            if cell['exp']=='IDEAL':
                cell['exp']='OBSTACLE'

    def pass_ft_param(self, x, y):
        """
        Update f(t) state when passing through a cell after obstacle detection.
        Depending on movement direction and previous exp state,
        mark EXPLOREING or CHECKING and reset f(t) timers.
        """
        # Determine movement direction for scanning
        dx, dy = self.last_move_dir
        # Horizontal pass
        if dx and not dy:
            for offset in range(-3,4):
                xx, yy = x, y+offset
                state = self.exp_grid[yy,xx]['exp']
                explored = Robot.grid_map.shared_ft_grid[yy,xx]['explored']
                if not explored:
                    if state=='IDEAL':
                        self.exp_grid[yy,xx]['exp']='EXPLOREING'
                        fcell = Robot.grid_map.shared_ft_grid[yy,xx]
                        fcell.update({'t':0,'k':0,'explored':True})
                    elif state in ('OBSTACLE','AVOID'):
                        self.exp_grid[yy,xx]['exp']='CHECKING'
                        fcell = Robot.grid_map.shared_ft_grid[yy,xx]
                        fcell.update({'t':0,'k':0,'explored':True})

    def end_ft_param(self):
        """
        Provide feedback to lambda values based on journey time upon completion.
        Adjust lambda for cells based on experience state and performance.
        """
        alpha = 1.01
        extra = min(self.journey_time/76, 1.5)
        for y in range(Robot.grid_map.height):
            for x in range(Robot.grid_map.width):
                state = self.exp_grid[y,x]['exp']
                fcell = Robot.grid_map.shared_ft_grid[y,x]
                if state=='OBSTACLE':
                    self.exp_grid[y,x]['exp']='IDEAL'
                elif state=='AVOID' or state=='CHECKING':
                    if extra>1:
                        fcell['lambda'] *= (2-extra)
                    self.exp_grid[y,x]['exp']='IDEAL'
                elif state=='EXPLOREING':
                    if self.journey_time<=76:
                        fcell['lambda'] *= alpha
                    else:
                        fcell['lambda'] *= (2-extra)
                    self.exp_grid[y,x]['exp']='IDEAL'
