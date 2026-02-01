# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 21:32:57 2026

@author: dsotn
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

# Constants
mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)

class PointWire:
    """Class representing a long straight wire as a point in 2D"""
    def __init__(self, x, y, current, direction='up'):
        """
        Initialize a wire as a point
        
        Parameters:
        x, y: coordinates of the wire (m)
        current: current magnitude (A), positive or negative
        direction: 'up' (+y), 'down' (-y), 'right' (+x), 'left' (-x)
        """
        self.position = np.array([x, y])
        self.current = current  # Can be positive or negative
        
        # Define direction vector based on direction string
        direction_vectors = {
            'up': np.array([0, 1]),
            'down': np.array([0, -1]),
            'right': np.array([1, 0]),
            'left': np.array([-1, 0])
        }
        
        if direction not in direction_vectors:
            raise ValueError(f"Direction must be one of: {list(direction_vectors.keys())}")
        
        self.direction_vector = direction_vectors[direction]
        
    def magnetic_field_at_point(self, point):
        """
        Calculate magnetic field at a given point due to this wire
        
        Parameters:
        point: (x, y) coordinates of the point where field is calculated (m)
        
        Returns:
        B: magnetic field vector (Bx, By) in Tesla
        """
        # For an infinite wire: B = (μ0*I)/(2πr) * φ_hat
        # where φ_hat = (-sin(φ), cos(φ)) = (-(y-y0)/r, (x-x0)/r) for current in +z direction
        
        r_vec = np.array(point) - self.position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-12:  # Avoid division by zero at wire location
            return np.array([0.0, 0.0])
        
        # Magnetic field magnitude
        B_magnitude = (mu0 * abs(self.current)) / (2 * np.pi * r)
        
        # Unit vector in azimuthal direction
        # For current in +z direction (out of page), B is clockwise
        # For 2D with current as scalar, we need to determine sign
        phi_hat = np.array([-r_vec[1], r_vec[0]]) / r
        
        # Adjust sign based on current direction and orientation
        # Current direction affects the sign of the field
        current_sign = np.sign(self.current)
        
        # If current is in -z direction (into page), field direction reverses
        # In our 2D representation, we'll use right-hand rule:
        # Current direction vector crossed with position vector gives B direction
        current_vec_3d = np.array([self.direction_vector[0], self.direction_vector[1], 0])
        r_vec_3d = np.array([r_vec[0], r_vec[1], 0])
        
        # Calculate cross product for direction
        cross_result = np.cross(current_vec_3d, r_vec_3d)
        B_direction = np.sign(cross_result[2])  # z-component of cross product
        
        B_vector = B_magnitude * current_sign * B_direction * phi_hat
        
        return B_vector
    
    def get_color(self):
        """Get color based on current direction and magnitude"""
        if self.current > 0:
            # Positive current: red shades
            intensity = min(abs(self.current) / 5.0, 1.0)
            return (0.8 + 0.2 * intensity, 0.2, 0.2)
        else:
            # Negative current: blue shades
            intensity = min(abs(self.current) / 5.0, 1.0)
            return (0.2, 0.2, 0.8 + 0.2 * intensity)

class TwoWireSystem2D:
    """Class representing a system of two parallel long wires in 2D"""
    def __init__(self, wire1, wire2):
        self.wire1 = wire1
        self.wire2 = wire2
        
    def total_magnetic_field(self, point):
        """
        Calculate total magnetic field at a point due to both wires
        
        Parameters:
        point: (x, y) coordinates where field is calculated
        
        Returns:
        B: total magnetic field vector (Bx, By) in Tesla
        """
        B1 = self.wire1.magnetic_field_at_point(point)
        B2 = self.wire2.magnetic_field_at_point(point)
        return B1 + B2
    
    def calculate_field_grid(self, x_lim=(-2, 2), y_lim=(-2, 2), grid_size=50):
        """
        Calculate magnetic field on a grid
        
        Returns:
        X, Y, Bx, By, B_magnitude: meshgrid and field components
        """
        x = np.linspace(x_lim[0], x_lim[1], grid_size)
        y = np.linspace(y_lim[0], y_lim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        Bx = np.zeros_like(X)
        By = np.zeros_like(X)
        B_magnitude = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                B = self.total_magnetic_field((X[i, j], Y[i, j]))
                Bx[i, j] = B[0]
                By[i, j] = B[1]
                B_magnitude[i, j] = np.linalg.norm(B)
        
        return X, Y, Bx, By, B_magnitude
    
    def plot_field_vectors(self, x_lim=(-2, 2), y_lim=(-2, 2), grid_size=15, scale=0.3):
        """Plot magnetic field vectors"""
        # Create grid for vector field
        x = np.linspace(x_lim[0], x_lim[1], grid_size)
        y = np.linspace(y_lim[0], y_lim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        
        # Calculate field at grid points
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                B = self.total_magnetic_field((X[i, j], Y[i, j]))
                Bx[i, j] = B[0]
                By[i, j] = B[1]
        
        # Normalize vectors for better visualization
        B_magnitude = np.sqrt(Bx**2 + By**2)
        mask = B_magnitude > 0
        Bx_norm = np.zeros_like(Bx)
        By_norm = np.zeros_like(By)
        Bx_norm[mask] = Bx[mask] / B_magnitude[mask]
        By_norm[mask] = By[mask] / B_magnitude[mask]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot vector field
        ax.quiver(X, Y, Bx_norm, By_norm, B_magnitude, 
                 cmap='hot', scale=1/scale, width=0.005, pivot='mid')
        
        # Plot wires as points with arrows indicating current direction
        self._plot_wires(ax, scale)
        
        ax.set_xlabel('X position (m)', fontsize=12)
        ax.set_ylabel('Y position (m)', fontsize=12)
        ax.set_title('Magnetic Field Vector Plot (2D)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        # Add colorbar for magnitude
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Field Magnitude (T)', fontsize=12)
        
        return fig, ax
    
    def plot_field_lines(self, x_lim=(-2, 2), y_lim=(-2, 2), density=2):
        """Plot magnetic field lines using streamlines"""
        # Create grid
        grid_size = 30
        x = np.linspace(x_lim[0], x_lim[1], grid_size)
        y = np.linspace(y_lim[0], y_lim[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        
        # Calculate field at grid points
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                B = self.total_magnetic_field((X[i, j], Y[i, j]))
                Bx[i, j] = B[0]
                By[i, j] = B[1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot streamlines
        ax.streamplot(X, Y, Bx, By, color='blue', 
                     density=density, linewidth=1.5, arrowsize=1)
        
        # Plot wires
        self._plot_wires(ax, scale=0.5)
        
        ax.set_xlabel('X position (m)', fontsize=12)
        ax.set_ylabel('Y position (m)', fontsize=12)
        ax.set_title('Magnetic Field Lines (2D Streamlines)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
        return fig, ax
    
    def plot_field_magnitude(self, x_lim=(-2, 2), y_lim=(-2, 2), grid_size=100):
        """Plot magnetic field magnitude as contour"""
        X, Y, Bx, By, B_magnitude = self.calculate_field_grid(x_lim, y_lim, grid_size)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Contour plot
        contour = ax1.contourf(X, Y, B_magnitude, levels=50, cmap='hot')
        plt.colorbar(contour, ax=ax1, label='Field Magnitude (T)')
        
        # Plot wires
        self._plot_wires(ax1, scale=0.5)
        
        ax1.set_xlabel('X position (m)', fontsize=12)
        ax1.set_ylabel('Y position (m)', fontsize=12)
        ax1.set_title('Magnetic Field Magnitude Contour', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 3D surface plot
        from mpl_toolkits.mplot3d import Axes3D
        ax3 = fig.add_subplot(122, projection='3d')
        surf = ax3.plot_surface(X, Y, B_magnitude, cmap='hot', 
                               alpha=0.8, linewidth=0, antialiased=True)
        
        # Plot wires as vertical lines
        ax3.scatter([self.wire1.position[0]], [self.wire1.position[1]], 
                   [np.max(B_magnitude)*1.1], color=self.wire1.get_color(), 
                   s=100, marker='o', label=f'Wire 1: {self.wire1.current}A')
        ax3.scatter([self.wire2.position[0]], [self.wire2.position[1]], 
                   [np.max(B_magnitude)*1.1], color=self.wire2.get_color(), 
                   s=100, marker='o', label=f'Wire 2: {self.wire2.current}A')
        
        ax3.set_xlabel('X position (m)', fontsize=12)
        ax3.set_ylabel('Y position (m)', fontsize=12)
        ax3.set_zlabel('Field Magnitude (T)', fontsize=12)
        ax3.set_title('3D Field Magnitude Surface', fontsize=14)
        ax3.legend()
        
        plt.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)
        
        return fig, (ax1, ax3)
    
    def _plot_wires(self, ax, scale=0.3):
        """Helper function to plot wires on an axis"""
        # Plot wire 1
        color1 = self.wire1.get_color()
        ax.scatter(self.wire1.position[0], self.wire1.position[1], 
                  color=color1, s=200, zorder=5, 
                  label=f'Wire 1: {self.wire1.current}A')
        
        # Add current direction arrow
        arrow_length = scale * 0.5
        ax.arrow(self.wire1.position[0], self.wire1.position[1],
                self.wire1.direction_vector[0] * arrow_length,
                self.wire1.direction_vector[1] * arrow_length,
                head_width=0.1, head_length=0.15, fc=color1, ec=color1)
        
        # Plot wire 2
        color2 = self.wire2.get_color()
        ax.scatter(self.wire2.position[0], self.wire2.position[1], 
                  color=color2, s=200, zorder=5,
                  label=f'Wire 2: {self.wire2.current}A')
        
        # Add current direction arrow
        ax.arrow(self.wire2.position[0], self.wire2.position[1],
                self.wire2.direction_vector[0] * arrow_length,
                self.wire2.direction_vector[1] * arrow_length,
                head_width=0.1, head_length=0.15, fc=color2, ec=color2)
        
        ax.legend(loc='upper right')
    
    def calculate_force_between_wires(self):
        """Calculate force per unit length between the two wires"""
        # Force per unit length between two parallel wires: F/L = (μ0 * I1 * I2) / (2π * d)
        r_vec = self.wire2.position - self.wire1.position
        d = np.linalg.norm(r_vec)
        
        if d < 1e-12:
            return np.array([0.0, 0.0])
        
        # Force magnitude per unit length
        force_magnitude = (mu0 * self.wire1.current * self.wire2.current) / (2 * np.pi * d)
        
        # Direction: attractive if currents are in same direction, repulsive if opposite
        # Unit vector from wire1 to wire2
        r_hat = r_vec / d
        
        # Force on wire2 due to wire1
        force_direction = np.sign(self.wire1.current * self.wire2.current)
        force_vector = force_magnitude * force_direction * r_hat
        
        return force_vector
    
    def analyze_special_points(self):
        """Analyze magnetic field at special points"""
        points = {
            'Midpoint': (self.wire1.position + self.wire2.position) / 2,
            'Wire 1': self.wire1.position,
            'Wire 2': self.wire2.position,
            'Above midpoint': (self.wire1.position + self.wire2.position) / 2 + np.array([0, 0.5]),
            'Right of wire 1': self.wire1.position + np.array([0.5, 0]),
        }
        
        print("\n" + "="*60)
        print("MAGNETIC FIELD ANALYSIS AT SPECIAL POINTS")
        print("="*60)
        
        for name, point in points.items():
            B = self.total_magnetic_field(point)
            B_mag = np.linalg.norm(B)
            print(f"{name:20s} at {point}: B = ({B[0]:.2e}, {B[1]:.2e}) T, |B| = {B_mag:.2e} T")

# Example configurations
def create_example_configurations():
    """Create example wire configurations"""
    examples = {}
    
    # Example 1: Parallel currents, same direction
    examples['Parallel Same'] = TwoWireSystem2D(
        PointWire(-0.5, 0, 2.0, 'up'),
        PointWire(0.5, 0, 2.0, 'up')
    )
    
    # Example 2: Parallel currents, opposite directions
    examples['Parallel Opposite'] = TwoWireSystem2D(
        PointWire(-0.5, 0, 2.0, 'up'),
        PointWire(0.5, 0, -2.0, 'down')  # Equivalent to -2.0 'up'
    )
    
    # Example 3: Different currents
    examples['Different Currents'] = TwoWireSystem2D(
        PointWire(-0.8, 0.2, 3.0, 'right'),
        PointWire(0.8, -0.2, -1.5, 'left')
    )
    
    # Example 4: Perpendicular currents
    examples['Perpendicular'] = TwoWireSystem2D(
        PointWire(-0.5, -0.5, 2.0, 'up'),
        PointWire(0.5, 0.5, 2.0, 'right')
    )
    
    return examples

def interactive_analysis():
    """Interactive analysis of two-wire systems"""
    print("="*60)
    print("INTERACTIVE ANALYSIS OF TWO LONG WIRES")
    print("="*60)
    
    # Get user input for wire configurations
    print("\nConfigure Wire 1:")
    x1 = float(input("  X position (m): ") or "-0.5")
    y1 = float(input("  Y position (m): ") or "0")
    I1 = float(input("  Current (A, positive/negative): ") or "2.0")
    dir1 = input("  Direction (up/down/left/right): ") or "up"
    
    print("\nConfigure Wire 2:")
    x2 = float(input("  X position (m): ") or "0.5")
    y2 = float(input("  Y position (m): ") or "0")
    I2 = float(input("  Current (A, positive/negative): ") or "2.0")
    dir2 = input("  Direction (up/down/left/right): ") or "up"
    
    # Create wires
    wire1 = PointWire(x1, y1, I1, dir1)
    wire2 = PointWire(x2, y2, I2, dir2)
    
    system = TwoWireSystem2D(wire1, wire2)
    
    # Analyze force between wires
    force = system.calculate_force_between_wires()
    force_magnitude = np.linalg.norm(force)
    print(f"\nForce per unit length between wires:")
    print(f"  Magnitude: {force_magnitude:.2e} N/m")
    print(f"  Direction: ({force[0]:.2e}, {force[1]:.2e})")
    
    if force_magnitude > 0:
        if np.dot(force, wire2.position - wire1.position) > 0:
            print("  Type: Repulsive")
        else:
            print("  Type: Attractive")
    
    # Analyze field at special points
    system.analyze_special_points()
    
    return system

if __name__ == "__main__":
    # Interactive mode or example mode
    choice = input("\nRun interactive mode? (y/n): ").lower()
    
    if choice == 'y':
        system = interactive_analysis()
        
        # Ask for visualization
        vis_choice = input("\nCreate visualizations? (y/n): ").lower()
        if vis_choice == 'y':
            fig1, ax1 = system.plot_field_vectors(x_lim=(-2, 2), y_lim=(-2, 2))
            fig2, ax2 = system.plot_field_lines(x_lim=(-2, 2), y_lim=(-2, 2))
            fig3, axes3 = system.plot_field_magnitude(x_lim=(-2, 2), y_lim=(-2, 2))
            plt.show()
    
    else:
        # Run example configurations
        examples = create_example_configurations()
        
        for name, system in examples.items():
            print(f"\n{'='*60}")
            print(f"EXAMPLE: {name}")
            print('='*60)
            
            # Calculate force between wires
            force = system.calculate_force_between_wires()
            force_mag = np.linalg.norm(force)
            print(f"Force between wires: {force_mag:.2e} N/m")
            
            # Create visualizations
            fig1, ax1 = system.plot_field_vectors(x_lim=(-1.5, 1.5), y_lim=(-1.5, 1.5))
            fig1.suptitle(f'{name} - Vector Field', fontsize=16)
            
            fig2, ax2 = system.plot_field_lines(x_lim=(-1.5, 1.5), y_lim=(-1.5, 1.5))
            fig2.suptitle(f'{name} - Field Lines', fontsize=16)
            
            fig3, axes3 = system.plot_field_magnitude(x_lim=(-1.5, 1.5), y_lim=(-1.5, 1.5))
            fig3.suptitle(f'{name} - Field Magnitude', fontsize=16)
        
        plt.show()