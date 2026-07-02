import os

import pandas as pd
import matplotlib.pyplot as plt

# CSV lives at the repo root (this file is under coex/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_REPO_ROOT, "mu_coex_comparison.csv")

# Read the CSV file, specifying comma as the delimiter
df = pd.read_csv(CSV_PATH, sep=',')

# Calculate y1 = -2 * epsilon
df['minus_2_epsilon'] = 2 * df['epsilon']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['epsilon'], df['minus_2_epsilon'], label='2 * epsilon', marker='o', linestyle='-')
plt.plot(df['epsilon'], df['mu_coex_SIM'], label='mu_coex_SIM', marker='x', linestyle='--')
plt.plot(df['epsilon'], df['mu_coex_FITTED'], label='mu_coex_FITTED', marker='s', linestyle=':')

# Add labels and title
plt.xlabel('epsilon')
plt.ylabel('Value')
plt.title('Comparison of SIM, FITTED, and -2*epsilon vs epsilon')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()