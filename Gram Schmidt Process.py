# Abhijeet Solanki
# Assignment 1
# Date: 08/26/2023

# Problem 4:
# Gram Schmidt Process
# Given a set of vectors v1 and v2, Use the Gram Schmidt process to orthornormalize them.

import numpy as np

# Given vectors
v1 = np.array([1, 1])
v2 = np.array([1, 2])

# Step 1: Compute u1
# To calculate the vector norm 
u1 = v1 / np.linalg.norm(v1)

# Step 2 and 3: Compute u2
v2_u1 = np.dot(v2, u1)
# To compute w2
w2 = v2 - (v2_u1)*u1
u2 = w2 / np.linalg.norm(w2)

# Display orthonormal vectors
print("Orthonormal Vectors u1 and u2 Using Gram Schmidt Process:")
print("u1:", u1)
print("u2:", u2)

# Output:
# (base) chiefaj@Abhijeets-MacBook-Pro CSC-6903 % /Users/chiefaj/miniconda3/envs/.env/bin/python "/Users/chiefaj/Tntech-Masters/CSC-6903/Gram Schmidt Process.py"
# Orthonormal Vectors u1 and u2 Using Gram Schmidt Process:
# u1: [0.70710678 0.70710678]
# u2: [-0.70710678  0.70710678]