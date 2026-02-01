# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 22:20:33 2026

@author: dsotn
"""

import numpy as np

def magnetic_field_rectangular_conductor(rect_corner, width, height, total_current, 
                                         field_point, n_divisions=50, mu_0=4*np.pi*1e-7):
    """
    Calculate magnetic field from a rectangular conductor with uniform current distribution.
    The conductor is treated as many parallel wires (current perpendicular to the plane).
    
    Parameters:
    -----------
    rect_corner : array-like [x, y]
        Coordinates of the left-bottom corner of rectangle
    width : float
        Width of rectangle (x-direction)
    height : float
        Height of rectangle (y-direction)
    total_current : float
        Total current through the rectangular cross-section (A)
        Positive = out of page
    field_point : array-like [x0, y0]
        Point where magnetic field is calculated
    n_divisions : int
        Number of divisions in each direction (total elements = n_divisions^2)
    mu_0 : float
        Permeability of free space
    
    Returns:
    --------
    Bx, By : float
        Magnetic field components at field_point
    """
    rect_corner = np.array(rect_corner)
    field_point = np.array(field_point)
    
    # Area of rectangle
    area = width * height
    
    # Current density (A/m²)
    J = total_current / area
    
    # Size of each small element
    dx = width / n_divisions
    dy = height / n_divisions
    
    # Area of each small element
    dA = dx * dy
    
    # Current in each small element
    dI = J * dA
    
    # Initialize total field
    Bx_total = 0.0
    By_total = 0.0
    
    # Create grid of element centers
    x_elements = rect_corner[0] + dx/2 + np.arange(n_divisions) * dx
    y_elements = rect_corner[1] + dy/2 + np.arange(n_divisions) * dy
    
    # Loop over all elements
    for x_elem in x_elements:
        for y_elem in y_elements:
            # Vector from element to field point
            dx_vec = field_point[0] - x_elem
            dy_vec = field_point[1] - y_elem
            
            # Distance from element to field point
            r = np.sqrt(dx_vec**2 + dy_vec**2)
            
            # Skip if field point is at element location
            if r < 1e-15:
                continue
            
            # Magnitude of B field from this element (like an infinitesimal wire)
            B_mag = (mu_0 * dI) / (2 * np.pi * r)
            
            # Direction: perpendicular to position vector (rotated 90° counterclockwise)
            dBx = -B_mag * dy_vec / r
            dBy = B_mag * dx_vec / r
            
            Bx_total += dBx
            By_total += dBy
    
    return Bx_total, By_total


# Example: Rectangular conductor
rect_corner = [0.1, 0.05]  # Left-bottom corner
width = 0.02               # Width (x-direction)
height = 0.03              # Height (y-direction)
total_current = 1000.0     # Total current in Amperes
field_point = [0.05, 0.01] # Point where we calculate B field

# Calculate magnetic field
Bx, By = magnetic_field_rectangular_conductor(
    rect_corner, width, height, total_current, field_point, n_divisions=50
)

B_magnitude = np.sqrt(Bx**2 + By**2)

print("Rectangular Conductor Parameters:")
print(f"  Corner: {rect_corner}")
print(f"  Width: {width} m")
print(f"  Height: {height} m")
print(f"  Total current: {total_current} A")
print(f"  Current density: {total_current/(width*height):.2f} A/m²")
print(f"\nField Point: {field_point}")
print(f"\nMagnetic Field Components:")
print(f"  Bx = {Bx:.6e} T")
print(f"  By = {By:.6e} T")
print(f"  |B| = {B_magnitude:.6e} T")

# Visualize the setup
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Draw rectangle
rect = Rectangle(rect_corner, width, height, 
                 linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5)
ax.add_patch(rect)

# Mark field point
ax.plot(field_point[0], field_point[1], 'ro', markersize=10, label='Field Point')

# Draw magnetic field vector (scaled for visibility)
scale = 0.02 / B_magnitude  # Scale factor for arrow
ax.arrow(field_point[0], field_point[1], Bx*scale, By*scale, 
         head_width=0.003, head_length=0.003, fc='red', ec='red', linewidth=2,
         label=f'B field (scaled)')

# Labels and formatting
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Rectangular Conductor and Magnetic Field', fontsize=14)
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend()

# Set reasonable axis limits
x_min = min(rect_corner[0], field_point[0]) - 0.02
x_max = max(rect_corner[0] + width, field_point[0]) + 0.02
y_min = min(rect_corner[1], field_point[1]) - 0.02
y_max = max(rect_corner[1] + height, field_point[1]) + 0.02
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('/home/claude/rectangular_conductor_field.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: rectangular_conductor_field.png")
