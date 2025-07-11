import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

# Load ratings data
ratings = pd.read_csv('rat1.csv')

# Convert ratings to Surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
model = SVD()
model.fit(trainset)

# Save the trained model
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as svd_model.pkl")
