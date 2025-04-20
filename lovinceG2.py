import numpy as np
import matplotlib.pyplot as plt
import os

def compute_Z_n(n_values, c=3e8, phi=1.618033988749895):
    """
    आपका सूत्र: Z_n = (9 * c * phi^n * pi^(3n-1) / 3^n) * exp(-i * n * pi / phi)
    n_values: n के मान (जैसे 1 से 9)।
    c: प्रकाश की गति (3e8 मीटर/सेकंड)।
    phi: स्वर्णिम अनुपात (1.618...)।
    """
    # त्रुटि जांच
    if phi == 0:
        raise ValueError("phi शून्य नहीं हो सकता")
    if not np.all(np.isfinite(n_values)):
        raise ValueError("n के मान वैध होने चाहिए")
    
    # ओवरफ्लो चेतावनी चालू करें
    np.seterr(all='warn')
    
    # परिमाण को लघुगणक में गणना करें (बड़े मानों के लिए)
    log_magnitude = (
        np.log(9 * c) +
        n_values * np.log(phi) +
        (3 * n_values - 1) * np.log(np.pi) -
        n_values * np.log(3)
    )
    
    # फेज की गणना
    phase = -n_values * np.pi / phi
    
    # परिमाण और फेज को मिलाएं
    Z_n = np.exp(log_magnitude) * np.exp(1j * phase)
    
    # गैर-वैध मानों की जांच
    if not np.all(np.isfinite(Z_n)):
        print("चेतावनी: Z_n में गैर-वैध मान मिले")
    
    return Z_n

def plot_Z_n(n_values, Z_n, save_plot=False):
    """
    Z_n का ग्राफ बनाता है: परिमाण, वास्तविक भाग, काल्पनिक भाग।
    save_plot: ग्राफ को फाइल में सेव करने के लिए।
    """
    plt.figure(figsize=(12, 7))
    plt.plot(n_values, np.abs(Z_n), label='|Z_n| (परिमाण)', marker='o', linewidth=2)
    plt.plot(n_values, np.real(Z_n), label='वास्तविक भाग', marker='s', linewidth=2)
    plt.plot(n_values, np.imag(Z_n), label='काल्पनिक भाग', marker='^', linewidth=2)
    plt.xlabel('n (अनुक्रम)', fontsize=12)
    plt.ylabel('मान', fontsize=12)
    plt.title('आपका सूत्र: Z_n जटिल अनुक्रम', fontsize=14)
    plt.yscale('symlog')  # बड़े मानों के लिए लघुगणकीय स्केल
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('Z_n_plot.png', dpi=300)
        print("ग्राफ 'Z_n_plot.png' के रूप में सेव किया गया")
    
    plt.show()

def save_results(n_values, Z_n, filename='Z_n_results.txt'):
    """
    परिणामों को फाइल में सेव करता है।
    """
    with open(filename, 'w') as f:
        f.write("n\tZ_n (वास्तविक + काल्पनिक)\t|Z_n|\n")
        for n, z in zip(n_values, Z_n):
            f.write(f"{n}\t{z:.4e}\t{np.abs(z):.4e}\n")
    print(f"परिणाम '{filename}' में सेव किए गए")

def main():
    # स्थिरांक
    c = 3e8  # प्रकाश की गति (m/s)
    phi = 1.618033988749895  # स्वर्णिम अनुपात
    n_values = np.arange(1, 10)  # n = 1 से 9
    
    # Z_n की गणना
    Z_n = compute_Z_n(n_values, c, phi)
    
    # परिणाम प्रिंट करें
    print("आपके सूत्र के Z_n मान:")
    for n, z in zip(n_values, Z_n):
        print(f"Z_{n} = {z:.4e} (परिमाण: {np.abs(z):.4e})")
    
    # परिणाम सेव करें
    save_results(n_values, Z_n)
    
    # ग्राफ बनाएं और सेव करें
    plot_Z_n(n_values, Z_n, save_plot=True)

if __name__ == "__main__":
    main()