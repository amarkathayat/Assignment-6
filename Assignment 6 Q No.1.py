import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Read the CSV file (ensure correct delimiter)
df = pd.read_csv("bank.csv", delimiter=";")

# Step 2: Select required columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

# Step 3: Convert categorical variables to dummy variables
df2['y'] = df2['y'].apply(lambda x: 1 if x == 'yes' else 0)
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# Step 4: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Define target variable and features
y = df3['y']
X = df3.drop(columns=['y'])

# Step 6: Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Logistic Regression Model
log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Step 8: Confusion Matrix & Accuracy for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix (Logistic Regression):\n", cm_log)
print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred_log):.4f}")

# Step 9: K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Step 10: Confusion Matrix & Accuracy for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (KNN):\n", cm_knn)
print(f"Accuracy (KNN): {accuracy_score(y_test, y_pred_knn):.4f}")

"""
Comparison of Models:
- Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}
- KNN Accuracy (k=3): {accuracy_score(y_test, y_pred_knn):.4f}

Observations:
- Logistic regression is generally better for large datasets with independent features.
- KNN performance depends on k-value; may not be ideal for high-dimensional data.
- Logistic regression is more interpretable and stable.
"""
