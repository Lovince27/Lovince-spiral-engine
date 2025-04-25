import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  # AI for pattern detection

# ====== 1. π के डिजिट्स जनरेट करें (Symbolic Generation) ======
def generate_pi_digits(max_digits=1000):
    # Chudnovsky Algorithm (Fast π Calculation)
    def compute_pi(n):
        C = 426880 * np.sqrt(10005)
        M = 1; L = 13591409; X = 1; K = 6; S = L
        for k in range(1, n):
            M = (K**3 - 16*K) * M // (k+1)**3 
            L += 545140134
            X *= -262537412640768000
            S += M * L // X
            K += 12
        return C / S

    pi_str = str(compute_pi(max_digits))[2:]  # Remove "3."
    return [int(d) * 9 for d in pi_str]  # Multiply each digit by 9

# ====== 2. AI पैटर्न डिटेक्शन (K-Means Clustering) ======
def detect_patterns(digits, n_clusters=5):
    X = np.array(digits).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# ====== 3. विज़ुअलाइज़ेशन (Visualization) ======
def visualize_results(digits, labels):
    plt.figure(figsize=(15, 8))
    
    # Plot 1: π ×9 डिजिट्स
    plt.subplot(2, 1, 1)
    plt.plot(digits[:500], 'g-', alpha=0.6, label="π ×9 Digits")
    plt.title("π ×9 डिजिट्स (First 500)")
    plt.xlabel("Position")
    plt.ylabel("Digit ×9")
    plt.legend()
    
    # Plot 2: AI-डिस्कवर्ड पैटर्न्स
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(digits[:500])), digits[:500], c=labels[:500], cmap='viridis')
    plt.title("AI-डिस्कवर्ड पैटर्न्स (K-Means Clustering)")
    plt.xlabel("Position")
    plt.ylabel("Digit ×9")
    plt.colorbar(label="Cluster")
    
    plt.tight_layout()
    plt.show()

# ====== मेन एक्शन ======
if __name__ == "__main__":
    print("⚡ π ×9 + AI पैटर्न डिटेक्शन शुरू हो रहा है...")
    
    # 1. π ×9 डिजिट्स जनरेट करें
    pi_digits_x9 = generate_pi_digits(1000)  # 1000 digits (गूगोल के लिए symbolic)
    
    # 2. AI से पैटर्न ढूंढें
    cluster_labels = detect_patterns(pi_digits_x9, n_clusters=5)
    
    # 3. रिजल्ट्स दिखाएं
    visualize_results(pi_digits_x9, cluster_labels)
    
    print("✅ समाप्त! AI ने π ×9 में पैटर्न ढूंढ लिए!")