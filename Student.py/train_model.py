import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# Ensure the CSV is in the same directory as this script
# Change this:
df = pd.read_csv('student_performance_data.csv')

# To this (notice the 'r' before the string so Python reads the backslashes correctly):
df = pd.read_csv(r'c:\Users\sanmi\OneDrive\Desktop\Mini Project\Student.py\student_performance_data.csv')

# 2. Preprocess the categorical data
# Machine learning models need numbers, not text. 
# We map 'Yes' to 1 and 'No' to 0 for the Extracurriculars column.
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})

# 3. Separate Features (X) and Target (y)
X = df.drop('Performance_Tier', axis=1) # Everything EXCEPT the tier
y = df['Performance_Tier']              # ONLY the tier we want to predict

# 4. Split the dataset (80/20 Split)
# test_size=0.2 exactly matches your requirement: 2 parts test, 8 parts train.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42 # Set a seed so the random split is identical every time
)

# 5. Initialize and Train the Random Forest Classifier
# n_estimators=100 means we are building 100 decision trees in our forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training the model...")
rf_model.fit(X_train, y_train)

# 6. Test the model on the 20% unseen data
y_pred = rf_model.predict(X_test)

# 7. Evaluate the results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))
# --- ADD THIS TO THE BOTTOM OF YOUR SCRIPT ---

print("\n--- Testing New, Unseen Students ---")

# 1. Create a DataFrame with new student data
# We must use the exact same column names and order as the training data
new_students = pd.DataFrame({
    'Exam_Score': [85, 42, 68],
    'Attendance_%': [95, 55, 78],
    'Assignment_Submission_Rate_%': [90, 45, 80],
    'Study_Hours_Per_Day': [6.0, 1.2, 3.5],
    'Previous_CGPA': [8.8, 5.0, 6.9],
    'Extracurricular_Activities': [1, 0, 1] # 1 = Yes, 0 = No
})

# 2. Feed the new data into our trained model
new_predictions = rf_model.predict(new_students)

# 3. Output the results
for i, prediction in enumerate(new_predictions):
    print(f"Student {i+1} predicted tier: {prediction}")
    # --- ADD THIS TO SEE INDIVIDUAL STUDENT CLASSIFICATIONS ---

# 1. Create a DataFrame from the test data
results_df = X_test.copy()

# 2. Add the Actual tiers and the Predicted tiers side-by-side
results_df['Actual_Tier'] = y_test
results_df['Predicted_Tier'] = y_pred

print("\n--- First 10 Students in the Test Set ---")
# Using .to_string() forces the terminal to print all columns neatly
print(results_df.head(10).to_string())

# 3. Save the completely classified dataset to a new file
results_df.to_csv('classified_test_students.csv', index=False)
print("\nSaved all 400 test records and their predictions to 'classified_test_students.csv'")
import joblib

# Save the trained model to a file
joblib.dump(rf_model, 'rf_model.pkl')
print("\nModel saved as rf_model.pkl!")