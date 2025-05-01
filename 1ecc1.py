import math

def fib(n):
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

def luc(n):
    a, b = 2, 1
    for _ in range(n): a, b = b, a + b
    return a

def pri(n):
    count, num = 0, 2
    while True:
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            count += 1
            if count == n:
                return num
        num += 1

def Spiral(n):
    result = [1]
    for i in range(2, n + 1):
        result.append(result[-1] + math.floor(math.sqrt(i)))
    return result[-1]

# Activation Functions
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x)))

# SuperState with activation
def evaluate_SuperState(n, psi_L=2.271, activation='gelu'):
    ΛC = Spiral(n)
    ΦS = (fib(n) * n**2) + pri(n)**(luc(n) % 5 + 1)
    raw_output = ΛC * ΦS * psi_L

    if activation == 'gelu':
        return gelu(raw_output)
    elif activation == 'mish':
        return mish(raw_output)
    else:
        return raw_output

# Demo runner
def run_demo(max_n=5, activation='gelu'):
    print(f"Evaluating SuperState(n) with {activation.upper()} activation for n = 1 to {max_n}")
    for n in range(1, max_n + 1):
        print(f"SuperState({n}) = {evaluate_SuperState(n, activation=activation)}")

if __name__ == "__main__":
    run_demo(activation='gelu')  # Try 'mish' also