# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 00:22:45 2026

@author: dsotn
"""

import math

def compute_integral(x0, a, y0, w):
    # Compute the four arctan terms
    term1 = (x0 + a) * math.atan((y0 + w) / (x0 + a))
    term2 = x0 * math.atan((y0 + w) / x0)
    term3 = (x0 + a) * math.atan(y0 / (x0 + a))
    term4 = x0 * math.atan(y0 / x0)
    
    # Compute the logarithmic terms
    log_term1 = ((y0 + w) / 2) * math.log(((x0 + a)**2 + (y0 + w)**2) / (x0**2 + (y0 + w)**2))
    log_term2 = (y0 / 2) * math.log(((x0 + a)**2 + y0**2) / (x0**2 + y0**2))
    
    # Combine all terms
    result = (term1 - term2 + log_term1) - (term3 - term4 + log_term2)
    return result

# Example usage
x0 = 1.0
a = 2.0
y0 = 3.0
w = 4.0
integral_value = compute_integral(x0, a, y0, w)
print(f"The value of the integral is: {integral_value}")