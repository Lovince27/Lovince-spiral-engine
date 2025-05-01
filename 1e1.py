# Creating the initial version of the Lovince Notation System (LNS) interpreter as lns_engine.py

lns_engine_code = '''
import math

# Fundamental sequences
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

# Evaluate SuperState(n)
def evaluate_SuperState(n, psi_L=2.271):
    ΛC = Spiral(n)
    ΦS = (fib(n) * n**2) + pri(n)**(luc(n) % 5 + 1)
    return ΛC * ΦS * psi_L

# Demo runner
def run_demo(max_n=5):
    print("Evaluating SuperState(n) for n = 1 to", max_n)
    for n in range(1, max_n + 1):
        print(f"SuperState({n}) = {evaluate_SuperState(n)}")

if __name__ == "__main__":
    run_demo()
'''

# Save to file
with open("/mnt/data/lns_engine.py", "w") as f:
    f.write(lns_engine_code)

"/mnt/data/lns_engine.py"
