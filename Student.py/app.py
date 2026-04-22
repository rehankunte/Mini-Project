import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from fpdf import FPDF

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
    'submission': fast_students['Assignment_Submission_%'].mean() if 'Assignment_Submission_%' in df.columns else fast_students.iloc[:, 3].mean() 
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
            pdf.cell(0, 6, f"   * {fail}", ln=True)
            
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
        if 'Assignment_Submission_%' in df.columns:
             input_data.rename(columns={'Assignment_Submission_Rate_%': 'Assignment_Submission_%'}, inplace=True)

        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        if prediction == "Fast":
            st.success(f"### Predicted Tier: {prediction} Learner 🚀\nKeep up the excellent work!")
        elif prediction == "Average":
            st.info(f"### Predicted Tier: {prediction} Learner 📚\nYou are doing well, but there is room to optimize.")
        else:
            st.warning(f"### Predicted Tier: {prediction} Learner ⚠️\nLet's look at how we can get you on track.")

        # The Advice Engine
        # --- THE ADVICE ENGINE ---
        failures_for_pdf = []
        if prediction in ["Slow", "Average"]:
            st.subheader("💡 Your Personalized Action Plan")
            
            if attendance < benchmarks['attendance']:
                msg = f"Attendance Alert ({attendance}% vs Top Average {benchmarks['attendance']:.1f}%): Missing lectures creates a compounding knowledge gap, as professors frequently drop hints about external exams during live sessions. Make a strict rule to attend all core lectures, and try sitting in the first three rows to force active engagement. Building a habit of showing up also connects you with high-performing peers who can help when concepts get tough."
                st.error(msg)
                failures_for_pdf.append(msg)
                
            if study < benchmarks['study']:
                msg = f"Study Volume Alert ({study} hrs vs Top Average {benchmarks['study']:.1f} hrs): Your current self-study time isn't quite enough to master complex concepts. To increase this without burning out, implement the Pomodoro Technique: work in highly focused 25-minute sprints followed by a 5-minute break. During these sprints, prioritize active recall and practice problems instead of just passively highlighting your textbook."
                st.error(msg)
                failures_for_pdf.append(msg)
                
            if submission < benchmarks['submission']:
                msg = f"Submission Consistency Alert ({submission}% vs Top Average {benchmarks['submission']:.1f}%): Skipping assignments leaves easy internal marks on the table and drastically lowers your baseline grade. Stop viewing assignments as massive hurdles; break them down into 15-minute micro-tasks the very day they are assigned. Consistently submitting work, even if it isn't completely perfect, builds a buffer that protects your overall GPA if you stumble on a main exam."
                st.error(msg)
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
    st.header("System Database")
    st.dataframe(df, use_container_width=True)