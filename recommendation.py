import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Step 1: Sample user-item ratings data
data = {
    'user': [1, 1, 1, 2, 2, 3, 3],
    'item': [1, 2, 3, 1, 3, 2, 3],
    'rating': [5, 3, 2, 4, 1, 2, 5]
}
df = pd.DataFrame(data)

# Step 2: Create user-item matrix
user_item_matrix = df.pivot(index='user', columns='item', values='rating').fillna(0)

# Step 3: Apply Truncated SVD (Matrix Factorization)
svd = TruncatedSVD(n_components=2)
matrix = user_item_matrix.values

# Decompose and reconstruct
U = svd.fit_transform(matrix)
Sigma = svd.singular_values_
VT = svd.components_

reconstructed_matrix = np.dot(U, VT)

# Step 4: Convert to DataFrame
predicted_ratings = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Step 5: Show output
print("\n🔢 Original Ratings Matrix:")
print(user_item_matrix)

print("\n🔮 Predicted Ratings Matrix:")
print(predicted_ratings.round(2))
