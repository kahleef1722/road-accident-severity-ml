import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create synthetic data matching 13 features
X = np.random.randint(0, 5, size=(200, 13))
y = np.random.choice([0,1,2], size=(200,))

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'accident_model.pkl')
print("Saved dummy model to accident_model.pkl")
