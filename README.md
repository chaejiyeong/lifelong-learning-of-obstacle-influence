# Lifelong Learning of Obstacle Influence

This repository provides supplementary source code for the published article:

> Jiyeong Chae, Sanghoon Lee, Hyunkyo Seo, and Kyung-Joon Park,
> "From issues to routes: A cooperative costmap with lifelong learning for Multi-AMR navigation,"
> Journal of Industrial Information Integration, vol. 48, Article 100941, 2025.
> DOI: https://doi.org/10.1016/j.jii.2025.100941

## Repository role

This repository contains the Python simulator used to evaluate the lifelong-learning mechanism for obstacle influence.
For the ROS 2 / Gazebo implementation of the cooperative costmap layer, see:
https://github.com/chaejiyeong/gazebo-simulator-for-cooperative-costmap

## License

This code is released under the Unlicense. See `LICENSE` for details.

# A Cooperative Costmap with Lifelong Learning for Multi-Robot Navigation

This repository accompanies the paper:

> **"A Cooperative Costmap with Lifelong Learning for Multi‑Robot Navigation"** 

It contains the source code for simulating a warehouse-like environment where multiple AMRs perform missions while collaboratively building and adapting a costmap over time.

---

## Project Structure

```bash
.
├── grid_map.py            # Grid map model and visualization
├── robot.py               # Robot behavior, navigation, and learning logic
├── obstacle.py            # Dynamic obstacle management
├── grid_map_widget.py     # PyQt5 GUI for simulation control and visualization
├── gui.py                 # Overall application GUI management
├── main.py                # Entry point to launch the simulation
├── README.md              # (You are here)
└── logs/                  # Folder where mission logs and learned lambda values are saved
```

---

## Key Features

- **Cooperative Perception**: Robots share obstacle observations to build a shared costmap.
- **Lifelong Learning**: Lambda values in the costmap are updated based on robot experience, improving navigation over time.
- **Dynamic Obstacles**: Obstacles appear and move across the map based on a Poisson process, simulating realistic warehouse traffic.
- **Mission Routing**: Robots are assigned missions (e.g., A→B→C→A) and dynamically replan when obstacles appear.
- **Real-time Visualization**: The PyQt5 GUI displays the environment map, shared perception map, f(t) decay curves, and mission statistics.
- **Flexible Simulation Speed**: Supports real-time and accelerated simulation via adjustable time scaling.

---

## Installation

**Requirements:**

- Python 3.8+
- PyQt5
- matplotlib
- numpy

**Install dependencies:**

```bash
pip install pyqt5 matplotlib numpy
```

---

## How to Run

Launch the simulation by executing:

```bash
python main.py
```

**Optional:**

- You can provide a suffix to distinguish saved log files:

```bash
python main.py simulation1
```

- Use `--grid-only` to launch a minimal view (only GridMap without full GUI):

```bash
python main.py --grid-only
```

---

## GUI Overview

| Component                  | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| **Grid Map View**           | Shows the actual environment with walls, dynamic obstacles, and robots. |
| **Shared Perception View**  | Displays robots' cooperative obstacle mapping and f(t)-based overlays.  |
| **Statistics Panel**        | Displays mission counts, route types, and completion rates.             |
| **Simulation Controls**     | Start/Stop, adjust time scale, and save mission logs.                    |
| **Static f(t) Graphs**       | Monitor decay behavior for specific key cells in real-time.             |

---

## Logging and Saving

During or after simulation:

- **Mission logs** (time, route type, journey time) are saved into CSV files inside the `logs/` folder.
- **Learned lambda values** for each cell are also stored automatically for post-analysis or reuse.

You can also **apply previously learned lambda values** to start the simulation with an already trained costmap.

---

## Learning Mechanism

- **Lambda Update**: When robots encounter obstacles or successfully pass through suspected areas, they adjust the decay rate (`lambda`) of f(t) accordingly.
- **Experience Grid**: Each robot internally tracks observations (IDEAL, OBSTACLE, AVOID, CHECKED, EXPLORED) to refine its model of the environment.

---

## Command-line Arguments

| Argument           | Type    | Description                                   | Example                         |
|--------------------|---------|-----------------------------------------------|---------------------------------|
| `suffix` (optional) | string  | Filename suffix for saving logs               | `python main.py test_run`       |
| `--grid-only`       | flag    | Launch GridMapWidget without full GUI controls| `python main.py --grid-only`    |

---


