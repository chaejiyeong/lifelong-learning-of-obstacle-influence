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
import numpy as np
import matplotlib
# Use Qt5Agg backend for Matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QSpinBox, QPushButton, QTabWidget,
    QGridLayout, QGroupBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from grid_map_widget import GridMapWidget
from robot import Robot


class ParameterSlider(QWidget):
    """
    Widget for adjusting a numerical parameter via a slider and label.

    Provides real-time display of the current value with specified
    decimal precision.
    """
    def __init__(self, label, min_value, max_value, default_value,
                 decimal_places=1, parent=None):
        """
        Initialize the ParameterSlider widget.

        Args:
            label (str): text for the parameter name label
            min_value (float): minimum slider value
            max_value (float): maximum slider value
            default_value (float): initial slider value
            decimal_places (int): number of decimal places to display
            parent (QWidget): optional parent widget
        """
        super().__init__(parent)
        self.decimal_places = decimal_places
        # Scale factor to convert float to int for QSlider
        self.scale_factor = 10 ** decimal_places

        # Horizontal layout for label, slider, and value display
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Parameter name label
        self.label = QLabel(label)
        layout.addWidget(self.label)

        # Slider setup
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(int(min_value * self.scale_factor))
        self.slider.setMaximum(int(max_value * self.scale_factor))
        self.slider.setValue(int(default_value * self.scale_factor))
        self.slider.setTickPosition(QSlider.TicksBelow)
        # Set tick interval to 1/10th of the range
        self.slider.setTickInterval(int((max_value - min_value) * self.scale_factor / 10))
        layout.addWidget(self.slider, 1)

        # Label to show current numeric value
        self.value_label = QLabel(f"{default_value:.{decimal_places}f}")
        layout.addWidget(self.value_label)

        # Connect slider movement to label update
        self.slider.valueChanged.connect(self._update_value_label)

    def _update_value_label(self, value):
        """
        Update the displayed numeric label when slider moves.

        Args:
            value (int): current slider integer value
        """
        real_value = value / self.scale_factor
        self.value_label.setText(f"{real_value:.{self.decimal_places}f}")

    def value(self):
        """
        Return the current slider value as a float.
        """
        return self.slider.value() / self.scale_factor

    def set_value(self, value):
        """
        Programmatically set the slider to a given float value.

        Args:
            value (float): desired slider value
        """
        self.slider.setValue(int(value * self.scale_factor))


class MatplotlibCanvas(FigureCanvas):
    """
    Qt widget that embeds a Matplotlib FigureCanvas.
    Allows Matplotlib plots within PyQt layouts.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the canvas with a new Matplotlib Figure.

        Args:
            parent (QWidget): parent widget
            width (float): width in inches
            height (float): height in inches
            dpi (int): resolution in dots per inch
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        # Single subplot for drawing
        self.axes = fig.add_subplot(111)

        super().__init__(fig)
        self.setParent(parent)

        # Expand to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class SimulationGUI(QMainWindow):
    """
    Main application window for multi-robot path planning simulation.

    Contains a tabbed interface with a grid map simulation view and
    a statistics summary.
    """
    def __init__(self, file_suffix=""):
        """
        Initialize the main GUI window and all child widgets.

        Args:
            file_suffix (str): optional suffix for log filenames
        """
        super().__init__()

        # Window title and size
        self.setWindowTitle("Multi-Robot Path Planning Simulation")
        self.resize(1200, 800)

        # Set up central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create tab widget
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # --- Grid Map Simulation Tab ---
        self.grid_map_widget = GridMapWidget(file_suffix=file_suffix)
        self.tabs.addTab(self.grid_map_widget, "Grid Map")
        
        # --- Statistics Tab ---
        self.stats_tab = QWidget()
        self.tabs.addTab(self.stats_tab, "Statistics")
        self.stats_layout = QVBoxLayout(self.stats_tab)

        # Title label for statistics section
        self.stats_title = QLabel("Mission Statistics")
        self.stats_title.setAlignment(Qt.AlignCenter)
        self.stats_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.stats_layout.addWidget(self.stats_title)

        # Table to display aggregated mission data
        self.mission_stats_table = QTableWidget(0, 3)
        self.mission_stats_table.setHorizontalHeaderLabels([
            "Route Type", "Journey Time", "Count"
        ])
        self.mission_stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.stats_layout.addWidget(self.mission_stats_table)

        # Button to refresh the statistics table
        self.update_mission_stats_button = QPushButton("Update Mission Statistics")
        self.update_mission_stats_button.clicked.connect(self.update_mission_stats)
        self.stats_layout.addWidget(self.update_mission_stats_button)

        # Run an initial simulation on startup
        self.run_simulation()

        # Update stats when user switches to the Statistics tab
        self.tabs.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        """
        Handle logic when user changes tabs.
        If switching to Statistics, refresh the table.

        Args:
            index (int): index of newly selected tab
        """
        tab_name = self.tabs.tabText(index)
        if tab_name == "Statistics":
            self.update_mission_stats()

    def update_mission_stats(self):
        """
        Populate the mission statistics table with data from Robot.mission_stats.
        Clears existing rows and inserts new ones.
        """
        self.mission_stats_table.setRowCount(0)
        for i, ((route_type, journey_time), count) in enumerate(Robot.mission_stats.items()):
            self.mission_stats_table.insertRow(i)
            # Display route type
            self.mission_stats_table.setItem(
                i, 0, QTableWidgetItem(str(route_type))
            )
            # Format journey time with two decimal places and 's' suffix
            self.mission_stats_table.setItem(
                i, 1, QTableWidgetItem(f"{journey_time:.2f}s")
            )
            # Display count
            self.mission_stats_table.setItem(
                i, 2, QTableWidgetItem(str(count))
            )

    def run_simulation(self):
        """
        Execute the simulation logic using current parameters.
        Prints a start message; actual simulation hooks into
        the GridMapWidget timer and Robot logic.
        """
        print("Simulation started.")
        # Placeholder: actual simulation code is executed within GridMapWidget


# Entry point when run as a standalone script
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = SimulationGUI()
    gui.show()
    sys.exit(app.exec_())
