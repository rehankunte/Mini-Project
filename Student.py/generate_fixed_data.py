import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_records = 2000

# 1. Generate random base stats 
exam = np.random.randint(35, 100, n_records)
cgpa = np.random.uniform(4.0, 10.0, n_records)

# Add overlap/noise so attendance isn't a perfect predictor
attendance = np.clip(np.random.normal(75, 15, n_records), 30, 100)
submission = np.clip(np.random.normal(70, 20, n_records), 20, 100)
study = np.clip(np.random.normal(3.5, 1.5, n_records), 0.5, 8.0)
extra = np.random.choice([0, 1], n_records)

# 2. Force the Tier to be decided by Academic Performance
# We create a hidden composite score (60% Exam, 40% CGPA)
composite_score = (exam * 0.6) + ((cgpa * 10) * 0.4)

tiers = []
for score in composite_score:
    if score >= 80:
        tiers.append('Fast')
    elif score >= 60:
        tiers.append('Average')
    else:
        tiers.append('Slow')

# 3. Save the balanced dataset
df = pd.DataFrame({
    'Exam_Score': exam,
    'Attendance_%': attendance.astype(int),
    'Assignment_Submission_%': submission.astype(int),
    'Study_Hours_Per_Day': np.round(study, 1),
    'Previous_CGPA': np.round(cgpa, 1),
    'Extracurricular_Activities': extra,
    'Performance_Tier': tiers
})

df.to_csv('student_performance_data.csv', index=False)
print("New balanced data generated! Run your training script now.")