import numpy as np
from scipy.stats import qmc
import pandas as pd

# Number of samples to generate
n_samples = 10

# Define parameter ranges (min, max) for l1, l2, l3
param_ranges = {
    'l1': (470, 670),
    'l2': (250, 450),
    'l3': (30, 230)
}

# Fixed values
b_fixed = 50
t_fixed = 20

# Set random seed for reproducibility (optional, remove if not needed)
seed = 42

# Create Latin Hypercube Sampler for 3 dimensions (l1, l2, l3)
sampler = qmc.LatinHypercube(d=3, seed=seed)
sample = sampler.random(n=n_samples)

# Scale samples to the actual parameter ranges
l_bounds = [param_ranges['l1'][0], param_ranges['l2'][0], param_ranges['l3'][0]]
u_bounds = [param_ranges['l1'][1], param_ranges['l2'][1], param_ranges['l3'][1]]
scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

# Round to integers (remove if you want decimal precision)
scaled_sample = np.round(scaled_sample).astype(int)

# Build dataframe in required format
df = pd.DataFrame({
    'l1': scaled_sample[:, 0],
    'b1': b_fixed,
    't1': t_fixed,
    'l2': scaled_sample[:, 1],
    'b2': b_fixed,
    't2': t_fixed,
    'l3': scaled_sample[:, 2],
    'b3': b_fixed,
    't3': t_fixed
})

# Save to CSV
output_path = 'doe.csv'
df.to_csv(output_path, index=False)

print(f"Generated {n_samples} samples and saved to {output_path}")
print(df.head())