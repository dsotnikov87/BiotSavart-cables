# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 14:17:54 2026

@author: dsotn
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)

class LongWire:
    """Class representing a long straight wire carrying current"""
    def __init__(self, position, current_direction, current):
        """
        Initialize a wire
        
        Parameters:
        position: (x, y) coordinates of the wire (m)
        current_direction: (dx, dy) unit vector indicating current direction
        current: current magnitude (A)
        """
        self.position = np.array(position)
        self.current_direction = np.array(current_direction) / np.linalg.norm(current_direction)
        self.current = current
    
    def magnetic_field_at_point(self, point):
        """
        Calculate magnetic field at a given point due to this wire
        
        Parameters:
        point: (x, y) coordinates of the point where field is calculated (m)
        
        Returns:
        Bz: z-component of magnetic field (Tesla)
        """
        # For an infinite wire, B = (μ0*I)/(2πr) in the azimuthal direction
        r_vec = np.array(point) - self.position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-12:  # Avoid division by zero
            return 0
        
        # Magnetic field magnitude
        B_magnitude = (mu0 * self.current) / (2 * np.pi * r)
        
        # Direction: perpendicular to r_vec and current direction (right-hand rule)
        # For 2D, with current in xy-plane, B is in z-direction
        # B direction = cross(current_direction, r_vec/r)
        cross_product = np.cross(np.append(self.current_direction, 0), 
                                 np.append(r_vec/r, 0))
        B_direction = cross_product[2]  # Only z-component matters in 2D
        
        return B_magnitude * B_direction
    
    def get_plot_data(self, length=2):
        """Get data for plotting the wire"""
        start = self.position - 0.5 * length * self.current_direction
        end = self.position + 0.5 * length * self.current_direction
        return start, end


class TwoWireSystem:
    """Class representing a system of two parallel long wires"""
    def __init__(self, wire1, wire2):
        self.wire1 = wire1
        self.wire2 = wire2
    
    def total_magnetic_field(self, point):
        """
        Calculate total magnetic field at a point due to both wires
        
        Parameters:
        point: (x, y) coordinates where field is calculated
        
        Returns:
        Bz: total z-component of magnetic field (Tesla)
        """
        B1 = self.wire1.magnetic_field_at_point(point)
        B2 = self.wire2.magnetic_field_at_point(point)
        return B1 + B2
    
    def calculate_field_grid(self, x_lim=(-2, 2), y_lim=(-2, 2), grid_size=50):
        """
        Calculate magnetic field on a grid
        
        Returns:
        X, Y, B: meshgrid and magnetic field values
        """
        x = np.linspace(x_lim[0], x_lim[1], grid_size)
        y = np.linspace(y_lim[0], y_lim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        B = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                B[i, j] = self.total_magnetic_field((X[i, j], Y[i, j]))
        
        return X, Y, B
    
    def plot_magnetic_field_2D(self, x_lim=(-2, 2), y_lim=(-2, 2), grid_size=30):
        """Create 2D visualization of magnetic field"""
        X, Y, B = self.calculate_field_grid(x_lim, y_lim, grid_size)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Contour plot
        ax1 = axes[0]
        contour = ax1.contourf(X, Y, B, levels=50, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax1, label='Magnetic Field Bz (T)')
        
        # Plot wires
        start1, end1 = self.wire1.get_plot_data()
        start2, end2 = self.wire2.get_plot_data()
        
        ax1.plot([start1[0], end1[0]], [start1[1], end1[1]], 'r-', linewidth=3, label='Wire 1')
        ax1.plot([start2[0], end2[0]], [start2[1], end2[1]], 'b-', linewidth=3, label='Wire 2')
        
        # Add current direction indicators
        ax1.arrow(start1[0], start1[1], 
                 0.3*self.wire1.current_direction[0], 0.3*self.wire1.current_direction[1],
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax1.arrow(start2[0], start2[1],
                 0.3*self.wire2.current_direction[0], 0.3*self.wire2.current_direction[1],
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        ax1.set_xlabel('X position (m)')
        ax1.set_ylabel('Y position (m)')
        ax1.set_title('Magnetic Field Contour (Two Long Wires)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Vector field plot (streamlines)
        ax2 = axes[1]
        
        # Calculate field vectors on a coarser grid for streamlines
        x_coarse = np.linspace(x_lim[0], x_lim[1], 20)
        y_coarse = np.linspace(y_lim[0], y_lim[1], 20)
        Xc, Yc = np.meshgrid(x_coarse, y_coarse)
        
        U = np.zeros_like(Xc)  # For 2D visualization, we'll show streamlines
        V = np.zeros_like(Yc)
        
        # Calculate field at each point for streamlines
        for i in range(Xc.shape[0]):
            for j in range(Xc.shape[1]):
                point = (Xc[i, j], Yc[i, j])
                # For streamlines, we need the field direction
                r1 = np.array(point) - self.wire1.position
                r2 = np.array(point) - self.wire2.position
                
                # Azimuthal direction for each wire
                if np.linalg.norm(r1) > 1e-12:
                    dir1 = np.array([-r1[1], r1[0]]) / np.linalg.norm(r1)
                    B1_dir = dir1 * self.wire1.current / np.linalg.norm(r1)
                else:
                    B1_dir = np.array([0, 0])
                
                if np.linalg.norm(r2) > 1e-12:
                    dir2 = np.array([-r2[1], r2[0]]) / np.linalg.norm(r2)
                    B2_dir = dir2 * self.wire2.current / np.linalg.norm(r2)
                else:
                    B2_dir = np.array([0, 0])
                
                # Total field direction
                B_dir = B1_dir + B2_dir
                U[i, j] = B_dir[0]
                V[i, j] = B_dir[1]
        
        # Plot streamlines
        ax2.streamplot(Xc, Yc, U, V, color='green', density=2, linewidth=1, arrowsize=1)
        
        # Plot wires
        ax2.plot([start1[0], end1[0]], [start1[1], end1[1]], 'r-', linewidth=3)
        ax2.plot([start2[0], end2[0]], [start2[1], end2[1]], 'b-', linewidth=3)
        
        ax2.set_xlabel('X position (m)')
        ax2.set_ylabel('Y position (m)')
        ax2.set_title('Magnetic Field Streamlines')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_field_along_line(self, line_points=100):
        """Plot magnetic field along a line between the wires"""
        # Create points along a line between the wires
        wire1_pos = self.wire1.position
        wire2_pos = self.wire2.position
        
        # Line perpendicular to line connecting wires
        mid_point = (wire1_pos + wire2_pos) / 2
        direction = wire2_pos - wire1_pos
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        # Points along perpendicular line
        distances = np.linspace(-2, 2, line_points)
        points = [mid_point + d * perpendicular for d in distances]
        
        # Calculate field at each point
        B_values = [self.total_magnetic_field(p) for p in points]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(distances, B_values, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Midpoint')
        
        # Mark wire positions
        d1 = np.dot(wire1_pos - mid_point, perpendicular)
        d2 = np.dot(wire2_pos - mid_point, perpendicular)
        ax.axvline(x=d1, color='r', linestyle=':', alpha=0.7, label='Wire 1 position')
        ax.axvline(x=d2, color='b', linestyle=':', alpha=0.7, label='Wire 2 position')
        
        ax.set_xlabel('Distance along perpendicular line (m)')
        ax.set_ylabel('Magnetic Field Bz (T)')
        ax.set_title('Magnetic Field Along Line Perpendicular to Wire Axis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax


def example_parallel_wires():
    """Example: Two parallel wires with currents in the same direction"""
    # Wire 1: at (-0.5, 0), current in +y direction
    wire1 = LongWire(position=(-0.5, 0), current_direction=(0, 1), current=1.0)
    
    # Wire 2: at (0.5, 0), current in +y direction
    wire2 = LongWire(position=(0.5, 0), current_direction=(0, 1), current=1.0)
    
    system = TwoWireSystem(wire1, wire2)
    
    print("Two parallel wires with currents in the same direction")
    print(f"Wire 1: position {wire1.position}, current {wire1.current} A")
    print(f"Wire 2: position {wire2.position}, current {wire2.current} A")
    
    # Test magnetic field at a point
    test_point = (0, 1)
    B_test = system.total_magnetic_field(test_point)
    print(f"\nMagnetic field at point {test_point}: {B_test:.2e} T")
    
    return system


def example_anti_parallel_wires():
    """Example: Two parallel wires with currents in opposite directions"""
    # Wire 1: at (-0.5, 0), current in +y direction
    wire1 = LongWire(position=(-0.5, 0), current_direction=(0, 1), current=1.0)
    
    # Wire 2: at (0.5, 0), current in -y direction
    wire2 = LongWire(position=(0.5, 0), current_direction=(0, -1), current=1.0)
    
    system = TwoWireSystem(wire1, wire2)
    
    print("\n\nTwo parallel wires with currents in opposite directions")
    print(f"Wire 1: position {wire1.position}, current {wire1.current} A")
    print(f"Wire 2: position {wire2.position}, current {wire2.current} A")
    
    # Test magnetic field at a point
    test_point = (0, 1)
    B_test = system.total_magnetic_field(test_point)
    print(f"\nMagnetic field at point {test_point}: {B_test:.2e} T")
    
    return system


if __name__ == "__main__":
    # Run examples
    print("="*60)
    print("BIOT-SAVART LAW FOR TWO LONG PARALLEL WIRES")
    print("="*60)
    
    # Example 1: Same direction currents
    system1 = example_parallel_wires()
    fig1, axes1 = system1.plot_magnetic_field_2D(x_lim=(-1.5, 1.5), y_lim=(-1.5, 1.5))
    fig1.suptitle("Two Parallel Wires - Currents in Same Direction", fontsize=14)
    
    # Example 2: Opposite direction currents
    system2 = example_anti_parallel_wires()
    fig2, axes2 = system2.plot_magnetic_field_2D(x_lim=(-1.5, 1.5), y_lim=(-1.5, 1.5))
    fig2.suptitle("Two Parallel Wires - Currents in Opposite Directions", fontsize=14)
    
    # Plot field along a line for both cases
    fig3, ax3 = system1.plot_field_along_line()
    fig4, ax4 = system2.plot_field_along_line()
    
    plt.show()