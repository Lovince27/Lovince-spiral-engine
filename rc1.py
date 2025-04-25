return self.omniverse_powers[-1] ** self._nested_power(depth - 1)

from sympy import symbols, pi, Rational, Function, Mul, Pow, simplify

# Define metaphysical quantum operators
L = Function('MassOperator')
M = Function('Supercomputer')
R_963 = Function('Tesla963')
S = Function('Spiral')
PiOp = Function('Pi')
PhiOp = Function('Phi')
B = Function('Biophoton')
Q = Function('NestedInfiniteCoherence')
C = Function('DivineHarmonic')
RE = Function('EternalOperator')

# Symbolic state vectors
initial_state = symbols('⟨Eternal_Infinite_Chaitanya|')
final_state = symbols('|Infinite_Eternal_Era⟩')

# Define symbolic operations (quantum product chain)
Ψ_supercomputer = Mul(
    initial_state,
    L('Mass'),
    M((1,3,9,27,81,'∞')),
    R_963(),
    S((0,3,6,9,10,'∞')),
    PiOp(pi),
    PhiOp('φ'),
    B('Biophoton'),
    Q('∞'),
    C(9*pi),
    RE(),
    final_state
)

# Display symbolic formula
print("Ψ_supercomputer =")
print(simplify(Ψ_supercomputer))