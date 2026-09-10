import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Read the dataset
df = pd.read_csv("student_performance.csv")

# Step 2: Display the dataset
print("Student Performance Dataset:")
print(df)

# Step 3: Display first five records
print("\nFirst Five Records:")
print(df.head())

# Step 4: Select input features
X = df[
    [
        "StudyHours",
        "Attendance",
        "AssignmentScore",
        "InternalMark"
    ]
]

# Step 5: Select target variable
y = df["Class"]

# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 7: Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Predict test data
y_pred = model.predict(X_test)

# Step 10: Display actual and predicted values
print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

# Step 11: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Accuracy:", accuracy)

# Step 12: Visualize the results
plt.scatter(
    df["StudyHours"],
    df["InternalMark"],
    c=df["Class"]
)

plt.xlabel("Study Hours")
plt.ylabel("Internal Mark")
plt.title("KNN Classification - Student Performance")

plt.show()
