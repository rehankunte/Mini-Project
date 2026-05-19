import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import random  # <--- Add this line right here!
from fpdf import FPDF
from database import SessionLocal, PredictionLog

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance & Advisory Dashboard")

# --- 1. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('student_performance_data.csv')

model = load_model()
df = load_data()

# Calculate "Fast Learner" benchmarks for the Advice Engine
fast_students = df[df['Performance_Tier'] == 'Fast']
benchmarks = {
    'attendance': fast_students['Attendance_%'].mean(),
    'study': fast_students['Study_Hours_Per_Day'].mean(),
    'submission': fast_students['Assignment_Submission_Rate_%'].mean()
}

# --- 2. PDF GENERATION LOGIC ---
def create_report_card(inputs, prediction, failures):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 18)
    pdf.cell(0, 10, "Student Performance Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, f"Assessed Tier: {prediction.upper()} LEARNER", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "1. Behavioral Metrics", ln=True)
    pdf.set_font("Helvetica", '', 11)
    pdf.cell(0, 6, f"   - Exam Score: {inputs['Exam_Score']}/100", ln=True)
    pdf.cell(0, 6, f"   - Attendance: {inputs['Attendance_%']}%", ln=True)
    pdf.cell(0, 6, f"   - Daily Study Time: {inputs['Study_Hours_Per_Day']} hours", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", 'B', 12)
    pdf.cell(0, 8, "2. Academic Action Plan", ln=True)
    pdf.set_font("Helvetica", '', 11)
    
    if not failures:
        pdf.cell(0, 6, "   Excellent trajectory. Keep up the current habits.", ln=True)
    else:
        pdf.cell(0, 6, "   ATTENTION REQUIRED. Focus on the following gaps:", ln=True)
        for fail in failures:
            # Swap out the unsupported em-dash for a standard hyphen
            safe_fail = fail.replace("—", "-")
            
            # Using multi_cell instead of cell so long text wraps properly!
            pdf.multi_cell(0, 6, f"   * {safe_fail}")
            pdf.ln(2)
    return bytes(pdf.output())

# --- 3. DASHBOARD TABS ---
tab_predict, tab_eda, tab_data = st.tabs(["🔮 Classify & Advise", "📈 Mega EDA Dashboard", "📊 Database"])

with tab_predict:
    st.header("Enter Student Details")
    
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            exam = st.number_input("Exam Score (0-100)", 0, 100, 50)
            attendance = st.number_input("Attendance % (0-100)", 0, 100, 60)
            submission = st.number_input("Assignment Submission % (0-100)", 0, 100, 55)
        with col2:
            study = st.number_input("Study Hours per Day", 0.0, 24.0, 2.0, 0.5)
            cgpa = st.number_input("Previous CGPA (0-10)", 0.0, 10.0, 6.0, 0.1)
            extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Tier & Get Advice", type="primary", use_container_width=True)

    if submitted:
        extra_val = 1 if extra == "Yes" else 0
        input_dict = {
            'Exam_Score': exam, 'Attendance_%': attendance, 
            'Assignment_Submission_Rate_%': submission, 'Study_Hours_Per_Day': study, 
            'Previous_CGPA': cgpa, 'Extracurricular_Activities': extra_val
        }
        
        # Ensure column names match the model exactly by using the dataframe's columns
        input_data = pd.DataFrame([input_dict])
      
        prediction = model.predict(input_data)[0]
        prediction = model.predict(input_data)[0]
        
        # --- NEW: SAVE TO SQL DATABASE ---
        try:
            db = SessionLocal()
            new_log = PredictionLog(
                exam_score=int(exam),
                attendance=int(attendance),
                submission=int(submission),
                study_hours=float(study),
                cgpa=float(cgpa),
                extracurricular=extra_val,
                predicted_tier=prediction
            )
            db.add(new_log)
            db.commit()
            db.close()
            st.toast("✅ Prediction successfully logged to secure SQL database.")
        except Exception as e:
            st.error(f"Database error: {e}")

        
        st.markdown("---")
        if prediction == "Fast":
            st.success(f"### Predicted Tier: {prediction} Learner 🚀\nKeep up the excellent work!")
        elif prediction == "Average":
            st.info(f"### Predicted Tier: {prediction} Learner 📚\nYou are doing well, but there is room to optimize.")
        else:
            st.warning(f"### Predicted Tier: {prediction} Learner ⚠️\nLet's look at how we can get you on track.")

        # The Advice Engine
        # --- THE ADVICE ENGINE ---
        
        # --- 4. DYNAMIC ADVICE ENGINE ---
        failures_for_pdf = []
        if prediction in ["Slow", "Average"]:
            st.subheader("💡 Your Personalized Action Plan")
            
            if attendance < benchmarks['attendance']:
                shortfall = int(benchmarks['attendance'] - attendance)
                att_tips = [
                    f"Attendance Alert: You are {shortfall}% below the top-tier average. Missing lectures creates compounding knowledge gaps. Try sitting in the first three rows to force active engagement.",
                    f"Action Required: Top students maintain {benchmarks['attendance']:.1f}% attendance. Skipping classes means missing unofficial hints professors drop about the final exam. Make a strict rule to attend all core lectures.",
                    f"Consistency Check: You need to boost attendance by {shortfall}% to match fast learners. Building a habit of showing up connects you with high-performing peers who can help when concepts get tough."
                ]
                msg = random.choice(att_tips)
                st.error(f"**📉 {msg}**")
                failures_for_pdf.append(msg)
                
            if study < benchmarks['study']:
                gap = round(benchmarks['study'] - study, 1)
                study_tips = [
                    f"Study Volume: You are studying {gap} hours less than the benchmark. Implement the Pomodoro Technique: work in highly focused 25-minute sprints followed by a 5-minute break.",
                    f"Study Strategy: Your self-study time is at {study} hrs. To reach the {benchmarks['study']:.1f} hr average without burning out, prioritize active recall and practice problems instead of passive reading.",
                    f"Time Management: Increase your daily study by {gap} hours. Break this down into two smaller blocks—one in the morning and one at night—focusing entirely on past exam papers."
                ]
                msg = random.choice(study_tips)
                st.error(f"**📉 {msg}**")
                failures_for_pdf.append(msg)
                
            if submission < benchmarks['submission']:
                sub_tips = [
                    f"Submission Consistency: You are leaving easy internal marks on the table. Break assignments down into 15-minute micro-tasks the very day they are assigned.",
                    f"Internal Marks Alert: Top students submit {benchmarks['submission']:.1f}% of work. Consistently submitting work, even if imperfect, builds a buffer that protects your overall GPA.",
                    f"Workflow Optimization: Stop viewing assignments as massive hurdles. Start the first draft within 24 hours of receiving the prompt to avoid last-minute panic."
                ]
                msg = random.choice(sub_tips)
                st.error(f"**📉 {msg}**")
                failures_for_pdf.append(msg)

        # PDF Download Button
        pdf_bytes = create_report_card(input_dict, prediction, failures_for_pdf)
        st.download_button(
            label="⬇️ Download Official Report Card (PDF)",
            data=pdf_bytes, file_name="Student_Report.pdf", mime="application/pdf", type="primary"
        )

with tab_eda:
    st.header("Comprehensive Data Visualization")
    st.write("Analyze the training data across 8 distinct statistical dimensions.")
    
    # Row 1: Histogram & Box Plot
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Histogram (Score Frequencies)")
        fig_hist = px.histogram(df, x="Exam_Score", color="Performance_Tier", nbins=20, title="Exam Score Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.subheader("2. Box Plot (Outlier Detection)")
        fig_box = px.box(df, x="Performance_Tier", y="Attendance_%", color="Performance_Tier", title="Attendance Spread by Tier")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    
    # Row 2: Density Plot & Scatter Plot
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("3. Density Plot (Concentration)")
        fig_density = px.density_contour(df, x="Study_Hours_Per_Day", y="Exam_Score", color="Performance_Tier", title="Study vs Score Density")
        st.plotly_chart(fig_density, use_container_width=True)
    with c4:
        st.subheader("4. Scatter Plot (Correlations)")
        fig_scatter = px.scatter(df, x="Study_Hours_Per_Day", y="Exam_Score", color="Performance_Tier", trendline="ols", title="Study Hours Impact")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    
    # Row 3: Bar Chart & Heat Map
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("5. Bar Chart (Averages)")
        avg_scores = df.groupby("Performance_Tier")["Exam_Score"].mean().reset_index()
        fig_bar = px.bar(avg_scores, x="Performance_Tier", y="Exam_Score", color="Performance_Tier", title="Average Exam Score by Tier")
        st.plotly_chart(fig_bar, use_container_width=True)
    with c6:
        st.subheader("6. Correlation Heat Map")
        # Filter only numbers so strings don't break the heatmap
        numeric_df = df.select_dtypes(include=np.number)
        fig_heat = px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlations")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    
    # Row 4: Line Chart & Stem-and-Leaf Plot
    c7, c8 = st.columns(2)
    with c7:
        st.subheader("7. Line Chart (Trend Analysis)")
        # Group by rounded study hours to make a clean trendline
        df['Rounded_Study'] = df['Study_Hours_Per_Day'].round(0)
        trend_df = df.groupby('Rounded_Study')['Exam_Score'].mean().reset_index()
        fig_line = px.line(trend_df, x="Rounded_Study", y="Exam_Score", markers=True, title="Score Trend over Study Hours")
        st.plotly_chart(fig_line, use_container_width=True)
    with c8:
        st.subheader("8. Stem-and-Leaf Plot")
        st.write("A text-based distribution of the first 50 exam scores.")
        # Generate Stem and Leaf using pure Python
        sample_scores = df['Exam_Score'].head(50).astype(int)
        stems = {}
        for score in sample_scores:
            stem, leaf = divmod(score, 10)
            stems.setdefault(stem, []).append(leaf)
            
        stem_leaf_output = "Stem | Leaf\n---- | ----\n"
        for stem in sorted(stems.keys()):
            leaves = "".join(map(str, sorted(stems[stem])))
            stem_leaf_output += f" {stem:2}  | {leaves}\n"
            
        st.code(stem_leaf_output, language="text")

with tab_data:
   with tab_data:
    st.header("📋 System Database")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Dataset")
        st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("Prediction Logs (SQL)")
        try:
            db = SessionLocal()
            logs = db.query(PredictionLog).order_by(
                PredictionLog.timestamp.desc()
            ).limit(50).all()
            db.close()
            
            if logs:
                log_data = [{
                    "Time": l.timestamp.strftime("%d-%b %H:%M"),
                    "Exam": l.exam_score,
                    "Attendance": l.attendance,
                    "Submission": l.submission,
                    "Study Hrs": l.study_hours,
                    "CGPA": l.cgpa,
                    "Extra": "Yes" if l.extracurricular else "No",
                    "Tier": l.predicted_tier
                } for l in logs]
                st.dataframe(log_data, use_container_width=True)
            else:
                st.info("No predictions logged yet. Use the Classify tab first.")
        except Exception as e:
            st.error(f"Could not load logs: {e}")