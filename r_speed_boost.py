@numba.jit(nopython=True)
   def _fast_fractal_calc(sequence):
       # Numba-optimized inner loop