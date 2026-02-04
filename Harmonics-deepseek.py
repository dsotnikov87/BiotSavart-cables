# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:36:42 2026

@author: dsotn
"""

import numpy as np
from typing import Tuple, Dict

def calculate_harmonics_from_field(
    r_ref: float,           # Reference radius in meters
    phi: np.ndarray,        # Angular positions in radians (0 to 2π)
    Bx: np.ndarray,         # Bx field components at each phi
    By: np.ndarray,         # By field components at each phi
    n_max: int = 20,        # Maximum multipole order to compute
    main_n: int = 1,        # Main multipole order (1 for dipole, 2 for quadrupole, etc.)
    normalize: bool = True  # Whether to normalize to main field
) -> Tuple[Dict, Dict]:
    """
    Calculate normal (Bn) and skew (An) harmonics from Bx, By field components.
    
    Parameters:
    -----------
    r_ref : float
        Reference radius in meters
    phi : np.ndarray
        Angular positions in radians (length M)
    Bx, By : np.ndarray
        Magnetic field components at each phi (length M)
    n_max : int
        Maximum multipole order to compute
    main_n : int
        Main multipole order (n=1 for dipole, n=2 for quadrupole, etc.)
    normalize : bool
        If True, normalize harmonics to main field (units = 10^4 * Bn/Bmain)
    
    Returns:
    --------
    Bn_dict, An_dict : dict
        Dictionaries with multipole order as keys and values in Tesla (if normalize=False)
        or normalized units (if normalize=True)
    """
    
    # Check inputs
    if len(phi) != len(Bx) or len(phi) != len(By):
        raise ValueError("phi, Bx, and By must have the same length")
    
    if len(phi) < 2 * n_max:
        print(f"Warning: Fewer points ({len(phi)}) than recommended (≥ {2*n_max})")
    
    M = len(phi)
    
    # Method 1: Direct Fourier analysis of tangential field (most common)
    # B_theta = -Bx * sin(phi) + By * cos(phi)
    B_theta = -Bx * np.sin(phi) + By * np.cos(phi)
    
    # Perform Fourier transform
    # Using numpy's FFT for efficiency
    fft_result = np.fft.fft(B_theta) / M
    
    # Get the harmonic coefficients
    # The coefficient for harmonic n is at index n in the FFT result
    # Note: FFT frequencies are: 0, 1, 2, ..., M/2, -M/2+1, ..., -1
    harmonics_complex = np.zeros(n_max + 1, dtype=complex)
    
    for n in range(1, n_max + 1):
        # Get the coefficient for frequency n
        if n < M // 2:
            coeff = fft_result[n]
        else:
            # For n > M/2, use negative frequencies
            coeff = fft_result[n - M]
        
        # Apply phase correction if needed
        # The standard formula: Bn - i*An = (1/2π)∫ B_theta(θ) e^{-inθ} dθ
        harmonics_complex[n] = coeff
    
    # Extract normal (Bn) and skew (An) components
    # Convention: Bn - i*An = coefficient
    Bn = np.real(harmonics_complex)
    An = -np.imag(harmonics_complex)  # Negative sign due to convention
    
    # Method 2: Alternative direct integration (slower but explicit)
    # Uncomment to verify results
    """
    Bn_direct = np.zeros(n_max + 1)
    An_direct = np.zeros(n_max + 1)
    for n in range(1, n_max + 1):
        integrand_real = B_theta * np.cos(n * phi)
        integrand_imag = B_theta * np.sin(n * phi)
        Bn_direct[n] = np.trapz(integrand_real, phi) / (2 * np.pi)
        An_direct[n] = -np.trapz(integrand_imag, phi) / (2 * np.pi)
    
    print("Verification - Difference between FFT and direct integration:")
    for n in range(1, min(10, n_max + 1)):
        print(f"n={n}: ΔBn={abs(Bn[n]-Bn_direct[n]):.2e}, ΔAn={abs(An[n]-An_direct[n]):.2e}")
    """
    
    # Create dictionaries
    Bn_dict = {}
    An_dict = {}
    
    # Get main field value for normalization
    if normalize:
        # The main field is the dominant harmonic
        B_main = Bn[main_n] if main_n <= n_max else Bn[1]
        
        if abs(B_main) < 1e-12:
            print(f"Warning: Main field B{main_n} is very small: {B_main:.2e} T")
            B_main = 1.0  # Avoid division by zero
        
        # Normalize: multiply by 10^4 * (R_ref)^{n-1} / B_main
        for n in range(1, n_max + 1):
            if normalize:
                # Normalized units: 10^4 * Bn/Bmain
                scale = (r_ref ** (n - 1)) / (r_ref ** (main_n - 1))
                Bn_dict[n] = (Bn[n] / B_main) * scale * 10000.0
                An_dict[n] = (An[n] / B_main) * scale * 10000.0
            else:
                Bn_dict[n] = Bn[n]
                An_dict[n] = An[n]
    else:
        for n in range(1, n_max + 1):
            Bn_dict[n] = Bn[n]
            An_dict[n] = An[n]
    
    return Bn_dict, An_dict


def calculate_field_from_harmonics(
    r: float,
    phi: np.ndarray,
    Bn_dict: Dict,
    An_dict: Dict,
    r_ref: float = 0.015
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct Bx and By fields from harmonic coefficients.
    
    Useful for verifying the calculation.
    """
    Bx_rec = np.zeros_like(phi, dtype=complex)
    By_rec = np.zeros_like(phi, dtype=complex)
    
    for n in Bn_dict.keys():
        if n in Bn_dict:
            Bn = Bn_dict[n]
            An = An_dict.get(n, 0.0)
            
            # Complex harmonic coefficient
            C = Bn + 1j * An
            
            # Scale by (r/r_ref)^(n-1)
            scale = (r / r_ref) ** (n - 1)
            
            # Complex coordinate: x + iy = r * e^(iφ)
            z_power = (r * np.exp(1j * phi) / r_ref) ** (n - 1)
            
            # Field reconstruction: By + iBx = Σ C_n * (z/r_ref)^(n-1)
            field_complex = C * scale * z_power
            
            By_rec += np.real(field_complex)
            Bx_rec += np.imag(field_complex)
    
    return np.real(Bx_rec), np.real(By_rec)


# Example usage with synthetic data
def example_calculation():
    """Create a synthetic field with known harmonics and analyze it."""
    
    # Parameters
    r_ref = 0.015  # 15 mm reference radius
    phi = np.linspace(0, 2 * np.pi, 361)[:-1]  # 0 to 360 degrees (exclude 360°)
    M = len(phi)
    
    # Create a synthetic field with known harmonics
    # Let's create a dipole (n=1) with some imperfections
    B_main = 1.0  # Tesla - main dipole field
    
    # Define some harmonic content (in Tesla, not normalized yet)
    harmonics_true = {
        1: 1.0,     # Main dipole
        2: 0.01,    # Small quadrupole (skew would be A2)
        3: 0.005,   # Small sextupole
        4: 0.002,   # Small octupole
    }
    skew_true = {
        1: 0.0,     # No skew dipole
        2: 0.008,   # Some skew quadrupole
        3: 0.003,   # Small skew sextupole
        4: 0.001,   # Small skew octupole
    }
    
    # Generate Bx and By from harmonics
    Bx_synth = np.zeros(M)
    By_synth = np.zeros(M)
    
    for n, Bn in harmonics_true.items():
        if n > 10:  # Limit for example
            continue
        
        An = skew_true.get(n, 0.0)
        
        # Field contribution from this harmonic
        # Using the formula: By + iBx = Σ (Bn + iAn) * (r/r_ref * e^(iφ))^(n-1)
        z = r_ref * np.exp(1j * phi)  # Complex coordinate on reference circle
        term = (Bn + 1j * An) * (z / r_ref) ** (n - 1)
        
        Bx_synth += np.imag(term)
        By_synth += np.real(term)
    
    # Add some random noise to simulate real data
    noise_level = 1e-4
    Bx_synth += np.random.normal(0, noise_level, M)
    By_synth += np.random.normal(0, noise_level, M)
    
    # Calculate harmonics from the synthetic field
    print("=" * 60)
    print("Harmonic Analysis of Synthetic Field")
    print("=" * 60)
    print(f"Reference radius: {r_ref*1000:.1f} mm")
    print(f"Number of points: {M}")
    print(f"True harmonics (Tesla at r={r_ref*1000:.1f} mm):")
    for n in sorted(harmonics_true.keys()):
        if n <= 6:
            print(f"  n={n}: B{n} = {harmonics_true[n]:.6f} T, A{n} = {skew_true.get(n, 0):.6f} T")
    
    Bn_calc, An_calc = calculate_harmonics_from_field(
        r_ref=r_ref,
        phi=phi,
        Bx=Bx_synth,
        By=By_synth,
        n_max=10,
        main_n=1,
        normalize=False  # Get raw values in Tesla first
    )
    
    print("\nCalculated harmonics (Tesla):")
    print("n     Bn [T]        An [T]        Error Bn      Error An")
    print("-" * 60)
    for n in sorted(Bn_calc.keys()):
        if n <= 6:
            true_Bn = harmonics_true.get(n, 0.0)
            true_An = skew_true.get(n, 0.0)
            err_Bn = abs(Bn_calc[n] - true_Bn)
            err_An = abs(An_calc[n] - true_An)
            print(f"{n:<2}   {Bn_calc[n]:+.6e}   {An_calc[n]:+.6e}   {err_Bn:.2e}   {err_An:.2e}")
    
    # Now calculate normalized units (what's typically reported)
    Bn_norm, An_norm = calculate_harmonics_from_field(
        r_ref=r_ref,
        phi=phi,
        Bx=Bx_synth,
        By=By_synth,
        n_max=10,
        main_n=1,
        normalize=True
    )
    
    print("\nNormalized harmonics (units at r_ref):")
    print("n     bn [units]    an [units]")
    print("-" * 60)
    for n in sorted(Bn_norm.keys()):
        if n <= 8:
            print(f"{n:<2}   {Bn_norm[n]:+10.2f}   {An_norm[n]:+10.2f}")
    
    # Verify by reconstructing the field
    Bx_rec, By_rec = calculate_field_from_harmonics(
        r=r_ref,
        phi=phi,
        Bn_dict=Bn_calc,
        An_dict=An_calc,
        r_ref=r_ref
    )
    
    # Calculate reconstruction error
    error_Bx = np.max(np.abs(Bx_rec - Bx_synth))
    error_By = np.max(np.abs(By_rec - By_synth))
    
    print(f"\nReconstruction error:")
    print(f"  Max Bx error: {error_Bx:.2e} T")
    print(f"  Max By error: {error_By:.2e} T")
    
    return Bn_calc, An_calc, Bn_norm, An_norm


# For direct use with your data
def analyze_your_data():
    """
    Template function for analyzing your own field data.
    Replace the example data with your actual measurements.
    """
    # Your parameters
    r_ref = 0.015  # 15 mm in meters
    phi = np.linspace(0, 2 * np.pi, 361)[:-1]  # Your angular points
    
    # Your measured field data (replace these with your actual data)
    # These should be arrays of the same length as phi
    Bx_measured = np.zeros_like(phi)  # Replace with your Bx data
    By_measured = np.zeros_like(phi)  # Replace with your By data
    
    # If you have Cartesian coordinates instead of polar
    # x = r_ref * np.cos(phi)
    # y = r_ref * np.sin(phi)
    # Then interpolate Bx, By at these points from your field map
    
    # Calculate harmonics
    Bn, An = calculate_harmonics_from_field(
        r_ref=r_ref,
        phi=phi,
        Bx=Bx_measured,
        By=By_measured,
        n_max=20,
        main_n=1,  # Change to 2 for quadrupole, 3 for sextupole, etc.
        normalize=True
    )
    
    # Print results
    print("Harmonic Analysis Results")
    print("=" * 60)
    print(f"Reference radius: {r_ref*1000:.1f} mm")
    print("n     bn [units]    an [units]")
    print("-" * 60)
    
    for n in sorted(Bn.keys()):
        if abs(Bn[n]) > 0.01 or abs(An[n]) > 0.01:  # Only show significant harmonics
            print(f"{n:<2}   {Bn[n]:+10.2f}   {An[n]:+10.2f}")
    
    return Bn, An


if __name__ == "__main__":
    # Run the example with synthetic data
    Bn, An, Bn_norm, An_norm = example_calculation()
    
    # To analyze your own data, call:
    # Bn, An = analyze_your_data()