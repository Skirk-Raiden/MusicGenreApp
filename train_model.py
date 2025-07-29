import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
data = pd.read_csv('features_30_sec.csv')

# 2. Prepare the data
data = data.drop(columns=['filename', 'length'])
genre_encoder = LabelEncoder()
data['label'] = genre_encoder.fit_transform(data['label'])

X = data.drop(columns=['label'])
y = data['label']

# 3. Scale the features and split data
# Using X.values avoids the feature name warning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values) 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train the KNN model
print("Training a fresh KNN model...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 5. Evaluate the new model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ New Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the new model, scaler, and encoder
joblib.dump(knn, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(genre_encoder, 'genre_encoder.pkl')

print("🚀 Model has been retrained and saved successfully!")