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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import csv
import random
import time
from matplotlib.patches import Circle
from obstacle import ObstacleManager
import math
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvas


def f(t, lam, k=0):
    """
    Compute the obstacle influence function f(t) = exp(-lam * (ln(t))^2) + k.
    Returns 0 at t=0 to avoid log(0) error.

    Args:
        t (float): elapsed time since obstacle detection
        lam (float): decay rate parameter
        k (float, optional): constant offset term (default=0)

    Returns:
        float: influence value, or 0 if t==0
    """
    if t == 0:
        return 0
    return math.exp(-lam * (math.log(t) ** 2)) + k


def df(t, lam):
    """
    Compute the derivative of the influence function df/dt.
    Returns 0 at t=0 to avoid log(0) error.

    Args:
        t (float): elapsed time
        lam (float): decay rate

    Returns:
        float: derivative value, or 0 if t==0
    """
    if t == 0:
        return 0
    return -2 * lam * math.log(t) * math.exp(-lam * (math.log(t) ** 2))


def get_cost_color(fval):
    """
    Determine overlay color based on influence value fval.

    Color mapping:
      - fval > 0.5: pure red (#FF0000)
      - 0.1 < fval <= 0.5: linear blend from blue (#0000FF) at 0.1 to red (#FF0000) at 0.5
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

    # High influence: red
    if fval > 0.5:
        return "#FF0000"
    # Medium influence: blend red-blue based on normalized t
    elif fval > 0.1:
        t = (fval - 0.1) / 0.4  # maps fval 0.1->0, 0.5->1
        R = int(t * 255)
        G = 0
        B = int((1 - t) * 255)
        return '#{:02X}{:02X}{:02X}'.format(R, G, B)
    # Low influence or invalid: white
    else:
        return "#FFFFFF"


class GridMap:
    """
    Class to create and manage a grid map environment,
    including dynamic obstacles, shared perception, and inflation layers.
    """
    # Default initial lambda for all cells
    Initial_lambda = 1

    def __init__(self, width=84, height=56):
        """
        Initialize grid map dimensions, data structures, and default state.

        Args:
            width (int): number of columns in the grid
            height (int): number of rows in the grid
        """
        self.width = width
        self.height = height
        # Reference to widget for callbacks (set by GridMapWidget)
        self.widget = None

        # 1. Base grid without obstacles (walls & mission points only)
        self.base_grid = np.zeros((height, width), dtype=object)
        # 2. Real-time environment grid (shows dynamic obstacles)
        self.real_grid = np.zeros((height, width), dtype=object)
        # 3. Shared perception grid for robots to share obstacle info
        self.shared_perception_grid = np.zeros((height, width), dtype=object)
        # 4. Influence (f(t)) grid with metadata per cell
        self.shared_ft_grid = np.empty((height, width), dtype=object)
        # 5. Inflation layer grid for real environment
        self.inflation_grid = np.zeros((height, width), dtype=object)

        # Initialize f(t) grid with default lambda and explored flag
        for i in range(height):
            for j in range(width):
                self.shared_ft_grid[i, j] = {
                    't': 0,             # elapsed time
                    'f': 0,             # influence function value
                    'k': 0.0,           # constant offset
                    'lambda': GridMap.Initial_lambda,
                    'explored': True    # whether cell was observed
                }

        # Numeric code for inflation layer in perception
        self.INFLATION_VALUE = 2
        # Map obstacle coordinates to obstacle IDs
        self.obstacle_cells = {}
        # Map obstacle IDs to list of cell coordinates
        self.obstacle_id_to_cells = {}

        self.mission_points = []  # list of tuples (x, y, type)
        self.perception_initialized = False

        # Color mapping for mission point types A-F
        self.mission_colors = {type_: '#FFFFFF' for type_ in ['A','B','C','D','E','F']}

        # Load default map configuration from CSV
        self._create_default_map()
        # Initialize obstacle manager
        self.obstacle_manager = ObstacleManager(self)

    def _create_default_map(self):
        """
        Load map layout from 'map.csv', adjust grid dimensions dynamically,
        and populate real_grid, base_grid, shared_ft_grid, and mission_points.
        On error, fall back to an empty map with walls around edges.
        """
        try:
            # Construct CSV path relative to project root
            csv_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'src', 'map.csv'
            )
            # Read entire CSV to determine size
            with open(csv_path, 'r') as f:
                content = f.read().replace(' ', '')
                lines = [l for l in content.split('\n') if l.strip()]
                # Determine max width from comma-separated cells
                max_width = max(len(line.split(',')) for line in lines)
                self.width = max_width
                self.height = len(lines)
                print(f"Map size set from CSV: width={self.width}, height={self.height}")

            # Reinitialize real_grid and shared_ft_grid to match new size
            self.real_grid = np.zeros((self.height, self.width), dtype=object)
            self.shared_ft_grid = np.empty((self.height, self.width), dtype=object)
            for i in range(self.height):
                for j in range(self.width):
                    self.shared_ft_grid[i, j] = {
                        't': 0, 'f': 0, 'k': 0.0,
                        'lambda': GridMap.Initial_lambda,
                        'explored': True
                    }
            # Parse CSV rows into grid_data list of lists
            with open(csv_path, 'r') as f:
                content = f.read().replace(' ', '')
                lines = [l for l in content.split('\n') if l.strip()]
                grid_data = []
                for row_idx, line in enumerate(lines):
                    cells = line.split(',')
                    row = []
                    for cell in cells:
                        cell = cell.strip()
                        # Empty => 0, integer => int, A-F => label, else => 0
                        if not cell:
                            row.append(0)
                        elif cell.isdigit() or (cell.startswith('-') and cell[1:].isdigit()):
                            row.append(int(cell))
                        elif cell in ['A','B','C','D','E','F']:
                            row.append(cell)
                        else:
                            row.append(0)
                    # Pad short rows with zeros
                    row += [0] * (self.width - len(row))
                    grid_data.append(row)
                # Pad missing rows
                while len(grid_data) < self.height:
                    grid_data.append([0]*self.width)
                self.real_grid = np.array(grid_data, dtype=object)
                # Base grid same as real_grid initially (no dynamic obs)
                self.base_grid = self.real_grid.copy()
                # Update inflation and perception
                self.update_inflation_grid()
                self.initialize_shared_perception()
                # Convert any -1 obstacle codes in base_grid to empty space
                self.base_grid[self.base_grid == -1] = 0
                # Collect mission points
                self.mission_points = [
                    (x,y,v) for y in range(self.height) for x,v in enumerate(self.real_grid[y])
                    if v in ['A','B','C','D','E','F']
                ]
        except Exception as e:
            print(f"Map loading error: {e}")
            # On error, build empty map with boundary walls and sample mission points
            self.real_grid = np.zeros((self.height, self.width), dtype=object)
            self.base_grid = np.zeros((self.height, self.width), dtype=object)
            # Add walls around perimeter
            for x in range(self.width):
                self.real_grid[0,x] = 1
                self.real_grid[self.height-1,x] = 1
            for y in range(self.height):
                self.real_grid[y,0] = 1
                self.real_grid[y,self.width-1] = 1
            # Default mission points for testing
            test_pts = [(8,12,'A'),(59,12,'B'),(59,5,'C'),(59,26,'D'),(8,26,'E'),(8,32,'F')]
            for x,y,t in test_pts:
                if 0<=x<self.width and 0<=y<self.height:
                    self.real_grid[y,x] = t
                    self.mission_points.append((x,y,t))
            self.base_grid = self.real_grid.copy()

    def get_grid(self):
        """
        Return a copy of the real environment grid (includes dynamic obstacles).
        """
        return self.real_grid.copy()

    def get_base_grid(self):
        """
        Return a copy of the static base grid (walls & mission points only).
        """
        return self.base_grid.copy()

    def initialize_shared_perception(self):
        """
        Initialize shared perception grid with walls from base_grid
        and apply inflation layer around walls. Only does so once.

        Returns:
            bool: True if initialized this call, False if already initialized.
        """
        if not self.perception_initialized:
            self.shared_perception_grid = np.zeros((self.height, self.width), dtype=int)
            # Copy walls (code=1) from base_grid
            wall_mask = (self.base_grid == 1)
            self.shared_perception_grid[wall_mask] = 1
            self.add_inflation_layer_to_shared()
            self.perception_initialized = True
            return True
        return False

    def add_inflation_layer_to_shared(self):
        """
        Add inflation cells around every wall in shared_perception_grid.
        Inflation cells have code self.INFLATION_VALUE.
        """
        if self.shared_perception_grid is None:
            return
        temp = self.shared_perception_grid.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.shared_perception_grid[y,x] == 1:  # wall
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            if dy==0 and dx==0:
                                continue
                            ny, nx = y+dy, x+dx
                            if 0<=nx<self.width and 0<=ny<self.height:
                                if temp[ny,nx] == 0:
                                    temp[ny,nx] = self.INFLATION_VALUE
        self.shared_perception_grid = temp

    def add_inflation_layer_to_obstacle(self, x, y):
        """
        Add inflation cells around a newly discovered obstacle at (x,y)
        in the shared perception grid.
        """
        if self.shared_perception_grid is None:
            return
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dy==0 and dx==0:
                    continue
                ny, nx = y+dy, x+dx
                if 0<=nx<self.width and 0<=ny<self.height:
                    if self.shared_perception_grid[ny,nx] == 0:
                        self.shared_perception_grid[ny,nx] = self.INFLATION_VALUE

    def update_obstacles(self, current_time):
        """
        Advance dynamic obstacles via ObstacleManager and return whether map changed.

        Note: real_grid updates are transient; shared perception retains seen obstacles.

        Args:
            current_time (float): current simulation time

        Returns:
            bool: True if any obstacles changed this step
        """
        return self.obstacle_manager.update(current_time)

    def update_shared_perception(self, x, y, value=-1):
        """
        Mark a cell (x,y) as obstacle (value=-1) in shared perception,
        unless it's a mission point. Also applies inflation.

        Args:
            x (int): column index
            y (int): row index
            value (int): marker for obstacle (default -1)

        Returns:
            bool: True if cell updated, False otherwise
        """
        if not (0<=x<self.width and 0<=y<self.height):
            return False
        real_val = self.real_grid[y,x]
        # Skip mission points (codes 2-6)
        if isinstance(real_val,int) and 2<=real_val<=6:
            return False
        # For other cells, set obstacle marker
        self.shared_perception_grid[y,x] = value
        # Add inflation layer around this obstacle
        if real_val == -1:
            self.add_inflation_layer_to_obstacle(x,y)
        return True

    def update_ft_grid(self):
        """
        Update influence grid shared_ft_grid based on current shared_perception_grid.
        For each obstacle cell (code -1), increment time t, recompute f(t).
        If f < 0.5, clear perception and reset that cell.
        Finally, reapply inflation around walls.
        """
        for y in range(self.height):
            for x in range(self.width):
                if self.shared_perception_grid[y,x] == -1:
                    cell = self.shared_ft_grid[y,x]
                    cell['t'] += 1
                    t = cell['t']
                    lam = cell['lambda']
                    k = cell['k']
                    fval = f(t, lam, k)
                    cell['f'] = fval
                    # If influence falls below threshold, clear perception
                    if fval < 0.5:
                        self.shared_perception_grid[y,x] = 0
                        self.remove_surrounding_shared_pereception(x,y)
                        cell['explored'] = False
                        cell['t'] = 0
        self.add_inflation_layer_to_shared()

    def plot(self, ax=None, robots=None):
        """
        Render the real environment grid with walls, dynamic obstacles,
        inflation zones, robot paths, and robot icons.

        Args:
            ax (matplotlib.axes.Axes): axes to draw on (creates new if None)
            robots (list): optional list of Robot instances to overlay

        Returns:
            matplotlib.axes.Axes: updated axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))
        ax.set_facecolor('#f8f8f8')
        # Convert real_grid values to numeric codes for colormap
        numeric = np.zeros((self.height,self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                val = self.real_grid[y,x]
                if val == 1:
                    numeric[y,x] = 1  # wall
                elif val == -1:
                    numeric[y,x] = 2  # dynamic obstacle
                elif val == self.INFLATION_VALUE:
                    numeric[y,x] = 3  # inflation
                # A-F and empty count as 0
        cmap = ListedColormap(['white', '#a0a0a0', '#ff6b6b', '#ffd700'])
        ax.imshow(numeric, cmap=cmap, origin='upper', vmin=0, vmax=3)
        # Overlay faint grid lines
        for x in range(self.width+1):
            ax.axvline(x-0.5, color='lightgrey', linewidth=0.1, alpha=0.3)
        for y in range(self.height+1):
            ax.axhline(y-0.5, color='lightgrey', linewidth=0.1, alpha=0.3)
        # Plot robot paths below icons
        if robots:
            for robot in robots:
                if not robot.route_completed and getattr(robot,'path',None):
                    px = [p[0] for p in robot.path[robot.path_index:]]
                    py = [p[1] for p in robot.path[robot.path_index:]]
                    color = 'r' if getattr(robot,'paths_different',False) else 'b'
                    ax.plot(px,py, '-', color=color, alpha=0.4, linewidth=2, zorder=4)
        # Draw mission points
        for x,y,pt in self.mission_points:
            ax.text(x,y,pt,ha='center',va='center',color='black',fontweight='bold',zorder=7)
        # Draw robots as dual circles
        if robots:
            for robot in robots:
                if not robot.route_completed:
                    x,y = robot.get_position()
                    changed = getattr(robot,'paths_different',False)
                    outer_c = '#ff8080' if changed else '#80b0ff'
                    inner_c = '#ff4040' if changed else '#4080ff'
                    ax.add_patch(Circle((x,y),0.65,color=outer_c,alpha=0.8,zorder=8))
                    ax.add_patch(Circle((x,y),0.45,color=inner_c,alpha=0.9,zorder=9))
                    ax.text(x,y,f"{robot.robot_id%100:02d}",ha='center',va='center',color='white',fontsize=8,zorder=10)
        ax.set_title('Warehouse Environment', fontsize=14, fontweight='bold')
        return ax

    def plot_shared_perception(self, ax=None):
        """
        Visualize the cooperative probabilistic costmap based
        on shared perception grid and influence overlays.

        Args:
            ax (matplotlib.axes.Axes): axes to draw on (new if None)

        Returns:
            matplotlib.axes.Axes: updated axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12,6))
        ax.set_facecolor('#f8f8f8')
        grid = np.zeros((self.height,self.width),dtype=int) if self.shared_perception_grid is None else self.shared_perception_grid.copy()
        cmap = ListedColormap(['#ff4040','white','#a6c8ff','#d8a6ff'])
        # Map codes -1->0, 0->1,1->2,2->3
        shifted = np.ones_like(grid)
        shifted[grid==-1]=0; shifted[grid==0]=1; shifted[grid==1]=2
        if hasattr(self,'INFLATION_VALUE'):
            shifted[grid==self.INFLATION_VALUE]=3
        ax.imshow(shifted,cmap=cmap,origin='upper',vmin=0,vmax=3)
        for x in range(self.width+1): ax.axvline(x-0.5,color='lightgrey',linewidth=0.1,alpha=0.3)
        for y in range(self.height+1): ax.axhline(y-0.5,color='lightgrey',linewidth=0.1,alpha=0.3)
        ax.set_title('Cooperative Probabilistic Costmap',fontsize=14,fontweight='bold')
        # Overlay cost colors where perception==obstacle
        overlay=np.zeros((self.height,self.width,4))
        for y in range(self.height):
            for x in range(self.width):
                if grid[y,x]==-1:
                    rgba=mcolors.to_rgba(get_cost_color(self.shared_ft_grid[y,x]['f']),alpha=0.8)
                    overlay[y,x,:]=rgba
        ax.imshow(overlay,origin='upper')
        return ax

    def remove_surrounding_shared_pereception(self, x, y):
        """
        Remove inflation cells around cleared obstacle at (x,y) in shared perception.
        """
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if dx==0 and dy==0: continue
                ny, nx = y+dy, x+dx
                if 0<=nx<self.width and 0<=ny<self.height:
                    if self.shared_perception_grid[ny,nx]==self.INFLATION_VALUE:
                        self.shared_perception_grid[ny,nx]=0

    def update_inflation_grid(self):
        """
        Compute inflation layer around walls and obstacles in real_grid.
        """
        temp = self.real_grid.copy()
        for y in range(self.height):
            for x in range(self.width):
                if self.real_grid[y,x] in [1,-1]:  # wall or dynamic obstacle
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            if dx==0 and dy==0: continue
                            ny, nx = y+dy, x+dx
                            if 0<=nx<self.width and 0<=ny<self.height:
                                if temp[ny,nx]==0:
                                    temp[ny,nx]=self.INFLATION_VALUE
        self.inflation_grid = temp
    
    def update_ft_graphs(self):
        """
        Update the f(t) graphs on the canvas layout.
        """
        canvas_graph = FigureCanvas(fig)
        self.static_info_layout.addWidget(canvas_graph)
        plt.close(fig)