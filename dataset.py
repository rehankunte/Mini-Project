import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_records = 2000

# Distribute tiers: 25% Slow, 50% Average, 25% Fast
tiers = np.random.choice(
    ['Slow', 'Average', 'Fast'], 
    size=n_records, 
    p=[0.25, 0.50, 0.25]
)

data = []

# Generate correlated features based on the assigned tier
for tier in tiers:
    if tier == 'Slow':
        attendance = np.random.randint(40, 76)
        study_hours = round(np.random.uniform(0.5, 3.5), 1)
        submission_rate = np.random.randint(30, 71)
        prev_cgpa = round(np.random.uniform(4.0, 6.5), 2)
        exam_score = np.random.randint(30, 65)
        extra = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        
    elif tier == 'Average':
        attendance = np.random.randint(70, 91)
        study_hours = round(np.random.uniform(2.0, 5.5), 1)
        submission_rate = np.random.randint(65, 91)
        prev_cgpa = round(np.random.uniform(6.0, 8.2), 2)
        exam_score = np.random.randint(55, 85)
        extra = np.random.choice(['Yes', 'No'], p=[0.5, 0.5])
        
    else: # Fast
        attendance = np.random.randint(85, 101)
        study_hours = round(np.random.uniform(4.0, 8.0), 1)
        submission_rate = np.random.randint(85, 101)
        prev_cgpa = round(np.random.uniform(7.5, 10.0), 2)
        exam_score = np.random.randint(75, 101)
        extra = np.random.choice(['Yes', 'No'], p=[0.6, 0.4])

    data.append([
        exam_score, attendance, submission_rate, 
        study_hours, prev_cgpa, extra, tier
    ])

# Create DataFrame
columns = [
    'Exam_Score', 'Attendance_%', 'Assignment_Submission_Rate_%', 
    'Study_Hours_Per_Day', 'Previous_CGPA', 'Extracurricular_Activities', 
    'Performance_Tier'
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('student_performance_data.csv', index=False)
print("Dataset successfully created and saved as 'student_performance_data.csv'!")