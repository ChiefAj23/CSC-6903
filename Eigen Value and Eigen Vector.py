# Abhijeet Solanki
# Assignment 1
# Date: 08/26/2023

# Problem 3:
# Eigen Value and Eigen Vector
# Given the covariance matrix of a dataset C compute the eigen values and eigenvectors. Which direction has the highest variance?

import numpy as np

# Given covariance matrix 
C = np.array([[3, 2], 
              [2, 3]])

# Compute eigenvalues and eigenvectors
Eigenvalues, Eigenvectors = np.linalg.eig(C)

# Print  Eigenvalues and Eigenvectors
print("Eigenvalues:", Eigenvalues)
print("Eigenvectors:", Eigenvectors)

# Find the index of the eigenvalue with the highest variance
max_variance_index = np.argmax(Eigenvalues)

# Direction with highest variance is the eigenvector corresponding to the eigenvalue with max variance
direction_highest_variance = Eigenvectors[:, max_variance_index]

# Display results
print("Direction with highest variance:", direction_highest_variance)

# Output:
# (base) chiefaj@Abhijeets-MacBook-Pro CSC-6903 % /Users/chiefaj/miniconda3/envs/.env/bin/python "/Users/chiefaj/Tntech-Masters/CSC-6903/Eigen Value and Eigen Vector.py"
# Eigenvalues: [5. 1.]
# Eigenvectors: [[ 0.70710678 -0.70710678]
#  [ 0.70710678  0.70710678]]
# Direction with highest variance: [0.70710678 0.70710678]
