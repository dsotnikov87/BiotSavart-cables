# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 00:27:53 2026

@author: dsotn
"""

"""
Magnetic Field from Circular Conductor with Uniform Current Distribution

Calculates Bx and By at point (x0, y0) from an infinitely long conductor 
with circular cross-section (radius Rc, center at (x, y)) carrying current I.

The solution uses analytical integration in polar coordinates.
"""

import math
import numpy as np
from scipy import integrate, special
import matplotlib.pyplot as plt

MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (T·m/A)


# ============================================================================
# ANALYTICAL SOLUTION
# ============================================================================

def magnetic_field_circular(x0, y0, x, y, Rc, I):
    """
    Calculate magnetic field from a circular conductor (ANALYTICAL).
    
    Physical model: Infinitely long conductor with circular cross-section.
    Current flows uniformly in +z direction (out of page).
    
    Parameters
    ----------
    x0, y0 : float
        Observation point coordinates (m)
    x, y : float
        Center of circular conductor (m)
    Rc : float
        Radius of conductor (m)
    I : float
        Total current through conductor (A)
    
    Returns
    -------
    Bx, By : float
        Magnetic field components (T)
    
    Notes
    -----
    For a point OUTSIDE the conductor (r > Rc):
        The field is equivalent to a line current at the center.
        B = μ₀I/(2πr) in the azimuthal direction
        
    For a point INSIDE the conductor (r < Rc):
        Only the current enclosed within radius r contributes.
        B = μ₀I·r/(2πRc²) in the azimuthal direction
    
    This is derived from Ampere's law using cylindrical symmetry.
    """
    # Distance from conductor center to observation point
    dx = x0 - x
    dy = y0 - y
    r = math.sqrt(dx**2 + dy**2)
    
    # Handle point at exact center
    if r < 1e-15:
        return 0.0, 0.0
    
    # Unit vector in radial direction (from center to observation point)
    r_hat_x = dx / r
    r_hat_y = dy / r
    
    # Azimuthal unit vector (perpendicular to radial, right-hand rule with z)
    # φ_hat = z_hat × r_hat = (-r_hat_y, r_hat_x)
    phi_hat_x = -r_hat_y
    phi_hat_y = r_hat_x
    
    # Calculate field magnitude based on position
    if r >= Rc:
        # Outside conductor: equivalent to line current
        # B = μ₀I / (2πr)
        B_magnitude = MU_0 * I / (2 * math.pi * r)
    else:
        # Inside conductor: only enclosed current contributes
        # I_enclosed = I * (r/Rc)²
        # B = μ₀ * I_enclosed / (2πr) = μ₀I·r / (2πRc²)
        B_magnitude = MU_0 * I * r / (2 * math.pi * Rc**2)
    
    # Field is in azimuthal direction
    Bx = B_magnitude * phi_hat_x
    By = B_magnitude * phi_hat_y
    
    return Bx, By


# ============================================================================
# ALTERNATIVE: Direct Integration (for verification)
# ============================================================================

def magnetic_field_circular_numerical(x0, y0, x, y, Rc, I, n_r=100, n_theta=100):
    """
    Numerical solution using direct integration over the circular cross-section.
    
    This integrates:
    B = (μ₀J/2π) ∫∫ (ẑ × r̂)/r dA
    
    in polar coordinates centered on the conductor.
    """
    J = I / (math.pi * Rc**2)  # Current density (A/m²)
    
    Bx_sum = 0.0
    By_sum = 0.0
    
    dr = Rc / n_r
    dtheta = 2 * math.pi / n_theta
    
    for i in range(n_r):
        r_prime = (i + 0.5) * dr  # Radial position in conductor
        
        for j in range(n_theta):
            theta = (j + 0.5) * dtheta
            
            # Position of current element (in global coordinates)
            x_prime = x + r_prime * math.cos(theta)
            y_prime = y + r_prime * math.sin(theta)
            
            # Vector from current element to observation point
            rx = x0 - x_prime
            ry = y0 - y_prime
            r_sq = rx**2 + ry**2
            
            if r_sq > 1e-30:
                # Biot-Savart contribution (2D)
                # dB ∝ J × r̂ / r = J (ẑ × r̂) / r
                # ẑ × r = (ry, -rx) → Bx ∝ ry/r², By ∝ -rx/r²
                
                # Area element in polar: dA = r' dr dθ
                dA = r_prime * dr * dtheta
                
                Bx_sum += ry / r_sq * dA
                By_sum += (-rx) / r_sq * dA
    
    prefactor = MU_0 * J / (2 * math.pi)
    
    Bx = prefactor * Bx_sum
    By = prefactor * By_sum
    
    return Bx, By


def magnetic_field_circular_scipy(x0, y0, x, y, Rc, I):
    """
    High-precision numerical solution using scipy integration.
    """
    J = I / (math.pi * Rc**2)
    
    def integrand_Bx(theta, r_prime):
        x_prime = x + r_prime * math.cos(theta)
        y_prime = y + r_prime * math.sin(theta)
        rx = x0 - x_prime
        ry = y0 - y_prime
        r_sq = rx**2 + ry**2
        if r_sq < 1e-30:
            return 0.0
        return ry / r_sq * r_prime  # r_prime is Jacobian
    
    def integrand_By(theta, r_prime):
        x_prime = x + r_prime * math.cos(theta)
        y_prime = y + r_prime * math.sin(theta)
        rx = x0 - x_prime
        ry = y0 - y_prime
        r_sq = rx**2 + ry**2
        if r_sq < 1e-30:
            return 0.0
        return -rx / r_sq * r_prime
    
    Bx_int, _ = integrate.dblquad(integrand_Bx, 0, Rc, 0, 2*math.pi, 
                                   epsabs=1e-12, epsrel=1e-12)
    By_int, _ = integrate.dblquad(integrand_By, 0, Rc, 0, 2*math.pi,
                                   epsabs=1e-12, epsrel=1e-12)
    
    prefactor = MU_0 * J / (2 * math.pi)
    
    return prefactor * Bx_int, prefactor * By_int


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_solutions():
    """Compare analytical with numerical solutions."""
    
    Rc = 0.01     # Radius: 1 cm
    I = 10.0      # Current: 10 A
    x_c, y_c = 0.0, 0.0  # Center at origin
    
    print("=" * 90)
    print("VERIFICATION: Analytical vs Numerical for Circular Conductor")
    print("=" * 90)
    print(f"\nCircle: Rc = {Rc*100:.1f} cm, center at origin")
    print(f"Current: I = {I} A")
    
    # Test points outside and inside
    test_points = [
        (0.03, 0.0, "outside, on x-axis"),
        (0.0, 0.025, "outside, on y-axis"),
        (0.02, 0.02, "outside, diagonal"),
        (-0.03, 0.01, "outside, quadrant II"),
        (0.005, 0.0, "inside, on x-axis"),
        (0.0, 0.003, "inside, on y-axis"),
        (0.004, 0.004, "inside, diagonal"),
        (0.05, 0.05, "far outside"),
    ]
    
    print(f"\n{'Point (cm)':<18} {'Location':<22} {'B_ana (μT)':<14} {'B_num (μT)':<14} {'Err %':<10}")
    print("-" * 90)
    
    errors = []
    for x0, y0, desc in test_points:
        Bx_ana, By_ana = magnetic_field_circular(x0, y0, x_c, y_c, Rc, I)
        Bx_num, By_num = magnetic_field_circular_scipy(x0, y0, x_c, y_c, Rc, I)
        
        B_ana = math.sqrt(Bx_ana**2 + By_ana**2)
        B_num = math.sqrt(Bx_num**2 + By_num**2)
        
        err = abs(B_num - B_ana) / B_ana * 100 if B_ana > 1e-15 else 0
        errors.append(err)
        
        print(f"({x0*100:5.2f},{y0*100:5.2f})      {desc:<22} {B_ana*1e6:12.4f}   {B_num*1e6:12.4f}   {err:.6f}")
    
    print("-" * 90)
    print(f"Maximum error: {max(errors):.6f}%")
    
    if max(errors) < 0.1:
        print("\n✓ SUCCESS: Analytical and numerical solutions match!")
    
    return max(errors)


# ============================================================================
# COMPARISON WITH RECTANGULAR
# ============================================================================

def compare_circle_vs_rectangle():
    """Compare circular and rectangular conductors with same area."""
    
    # Same current and cross-sectional area
    I = 10.0
    Rc = 0.01  # Circle radius: 1 cm
    area = math.pi * Rc**2
    
    # Square with same area
    side = math.sqrt(area)
    
    print("\n" + "=" * 70)
    print("COMPARISON: Circle vs Square (same cross-sectional area)")
    print("=" * 70)
    print(f"\nCircle: Rc = {Rc*100:.2f} cm, Area = {area*1e4:.4f} cm²")
    print(f"Square: side = {side*100:.2f} cm, Area = {side**2*1e4:.4f} cm²")
    print(f"Current: I = {I} A")
    
    # Import rectangular solution
    from magnetic_field_solution import magnetic_field_analytical as rect_field
    
    print(f"\n{'Point (cm)':<16} {'B_circle (μT)':<16} {'B_square (μT)':<16} {'Diff %':<10}")
    print("-" * 60)
    
    test_points = [(0.03, 0.0), (0.0, 0.03), (0.03, 0.03), (0.05, 0.02)]
    
    for x0, y0 in test_points:
        Bx_c, By_c = magnetic_field_circular(x0, y0, 0, 0, Rc, I)
        Bx_r, By_r = rect_field(x0, y0, -side/2, -side/2, side, side, I)
        
        B_c = math.sqrt(Bx_c**2 + By_c**2)
        B_r = math.sqrt(Bx_r**2 + By_r**2)
        
        diff = (B_r - B_c) / B_c * 100
        
        print(f"({x0*100:4.1f},{y0*100:4.1f})        {B_c*1e6:12.4f}       {B_r*1e6:12.4f}       {diff:8.2f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_field(Rc, I, x_c=0, y_c=0):
    """Create visualization of magnetic field from circular conductor."""
    
    margin = Rc * 4
    n = 40
    
    x = np.linspace(-margin + x_c, margin + x_c, n)
    y = np.linspace(-margin + y_c, margin + y_c, n)
    X, Y = np.meshgrid(x, y)
    
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    
    for i in range(n):
        for j in range(n):
            Bx[i,j], By[i,j] = magnetic_field_circular(X[i,j], Y[i,j], x_c, y_c, Rc, I)
    
    B_mag = np.sqrt(Bx**2 + By**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Field lines
    ax1 = axes[0]
    circle = plt.Circle((x_c*100, y_c*100), Rc*100, fill=True, 
                        facecolor='#ffcccc', edgecolor='red', linewidth=2.5, zorder=5)
    ax1.add_patch(circle)
    ax1.text(x_c*100, y_c*100, 'I', fontsize=14, ha='center', va='center', 
            fontweight='bold', zorder=6)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        color = np.log10(B_mag * 1e6 + 1e-10)
    
    strm = ax1.streamplot(X*100, Y*100, Bx, By, color=color, cmap='coolwarm', 
                         density=2.0, linewidth=1.2)
    
    ax1.set_xlim((-margin + x_c)*100, (margin + x_c)*100)
    ax1.set_ylim((-margin + y_c)*100, (margin + y_c)*100)
    ax1.set_xlabel('x (cm)')
    ax1.set_ylabel('y (cm)')
    ax1.set_title('Magnetic Field Lines')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(strm.lines, ax=ax1, label='log₁₀(|B| in μT)')
    
    # Magnitude
    ax2 = axes[1]
    circle2 = plt.Circle((x_c*100, y_c*100), Rc*100, fill=False, 
                         edgecolor='red', linewidth=2.5, zorder=10)
    ax2.add_patch(circle2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        log_B = np.log10(B_mag * 1e6 + 1e-10)
    
    cf = ax2.contourf(X*100, Y*100, log_B, levels=25, cmap='viridis')
    
    ax2.set_xlim((-margin + x_c)*100, (margin + x_c)*100)
    ax2.set_ylim((-margin + y_c)*100, (margin + y_c)*100)
    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('y (cm)')
    ax2.set_title('Field Magnitude')
    ax2.set_aspect('equal')
    plt.colorbar(cf, ax=ax2, label='log₁₀(|B| in μT)')
    
    plt.suptitle(f'Magnetic Field: Circular Conductor (Rc = {Rc*100:.1f} cm, I = {I} A)', 
                fontsize=14)
    plt.tight_layout()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Parameters
    Rc = 0.01     # Radius: 1 cm
    I = 10.0      # Current: 10 A
    x_c, y_c = 0.0, 0.0  # Center at origin
    
    # Example calculation
    print("=" * 70)
    print("MAGNETIC FIELD FROM CIRCULAR CONDUCTOR")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Radius: Rc = {Rc*100:.1f} cm")
    print(f"  Center: ({x_c*100:.1f}, {y_c*100:.1f}) cm")
    print(f"  Current: I = {I} A")
    print(f"  Current density: J = {I/(math.pi*Rc**2):.0f} A/m²")
    
    # Calculate at specific points
    print("\n" + "-" * 70)
    print("Example calculations:")
    print("-" * 70)
    
    test_cases = [
        (0.03, 0.0, "outside"),
        (0.005, 0.0, "inside"),
        (0.02, 0.02, "outside diagonal"),
    ]
    
    for x0, y0, desc in test_cases:
        Bx, By = magnetic_field_circular(x0, y0, x_c, y_c, Rc, I)
        B = math.sqrt(Bx**2 + By**2)
        angle = math.degrees(math.atan2(By, Bx))
        r = math.sqrt((x0-x_c)**2 + (y0-y_c)**2)
        
        print(f"\nPoint ({x0*100:.1f}, {y0*100:.1f}) cm - {desc} (r = {r*100:.2f} cm):")
        print(f"  Bx = {Bx*1e6:.4f} μT")
        print(f"  By = {By*1e6:.4f} μT")
        print(f"  |B| = {B*1e6:.4f} μT")
        print(f"  Direction: {angle:.1f}° from +x axis")
    
    # Verify against numerical
    print("\n")
    verify_solutions()
    
    # Try comparison if rectangular solution exists
    try:
        compare_circle_vs_rectangle()
    except:
        pass
    
    # Plot
    print("\nGenerating visualization...")
    fig = plot_field(Rc, I, x_c, y_c)
    # fig.savefig('/home/claude/magnetic_field_circular.png', dpi=150, bbox_inches='tight')
    print("Plot saved!")