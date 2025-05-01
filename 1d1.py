import numpy as np  

def lovince_formula(x, alpha=1, lambda_=1):  
    math_term = np.exp(1j * np.pi) + 1                  # Euler's identity  
    physics_term = alpha * (299792458 * 1.0545718e-34) / 6.67430e-11  
    ai_term = lambda_ * np.maximum(0, x)                 # ReLU  
    return math_term + physics_term + ai_term  

# Example:  
print(lovince_formula(x=2.0, alpha=0.1, lambda_=0.5))    # Output: (Complex result)  