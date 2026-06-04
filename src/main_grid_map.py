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
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from grid_map import GridMapWidget
from gui import SimulationGUI


class RobotSimulationApp(QMainWindow):
    """
    Main window class for the warehouse robot simulation app,
    displaying only the grid map widget if desired.
    """
    def __init__(self):
        """
        Initialize the main window, set title, size, and central widget.
        """
        super().__init__()
        # Window title shown in the OS title bar
        self.setWindowTitle("Warehouse Robot Simulation")
        # Initial window position (x=100, y=100) and size (width=1000, height=700)
        self.setGeometry(100, 100, 1000, 700)

        # Central widget: the grid map view for robot navigation
        self.grid_map_widget = GridMapWidget()
        self.setCentralWidget(self.grid_map_widget)


def main():
    """
    Entry point for the Robot Simulation application.

    Launches either the standalone grid map view or the
    full simulation GUI based on command-line arguments.
    """
    # Create the Qt application instance
    app = QApplication(sys.argv)

    # If '--grid-only' flag is provided, show only the GridMapWidget
    if len(sys.argv) > 1 and sys.argv[1] == '--grid-only':
        window = RobotSimulationApp()
    else:
        # Otherwise, launch the full-featured simulation GUI
        window = SimulationGUI()

    # Display the selected main window
    window.show()
    # Enter the Qt main event loop and exit cleanly on close
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
