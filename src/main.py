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

"""
Main entry point for the simulation application.
This module parses command-line arguments, initializes the Qt application,
and launches the SimulationGUI window.
"""
import sys
import argparse
from PyQt5.QtWidgets import QApplication
from gui import SimulationGUI


def main():
    """
    Main function that sets up argument parsing and starts the Qt event loop.
    
    The function reads an optional 'suffix' argument for file saving,
    prints the received suffix, initializes the QApplication,
    creates the SimulationGUI with the provided suffix,
    displays the window, and enters the Qt main loop.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Simulation Application"
    )
    # Optional positional argument for file suffix when saving logs or outputs
    parser.add_argument(
        'suffix',
        type=str,
        nargs='?',  # zero or one argument
        default='',
        help='Suffix to append to filenames when saving files'
    )
    args = parser.parse_args()

    # Echo the provided suffix for debugging purposes
    print(f"Received suffix: {args.suffix}")

    # Create the Qt application instance
    app = QApplication(sys.argv)
    # Initialize the main simulation GUI, passing along the suffix
    window = SimulationGUI(file_suffix=args.suffix)
    # Show the main window
    window.show()
    # Enter the Qt event loop; sys.exit ensures proper exit code propagation
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
