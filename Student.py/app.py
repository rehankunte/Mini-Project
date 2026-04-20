import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("🎓 Student Performance & Advisory Dashboard")

# 2. Load the saved model and dataset
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('student_performance_data.csv')

model = load_model()
df = load_data()

# Calculate "Fast Learner" benchmarks dynamically from the dataset
fast_students = df[df['Performance_Tier'] == 'Fast']
benchmarks = {
    'attendance': fast_students['Attendance_%'].mean(),
    'study': fast_students['Study_Hours_Per_Day'].mean(),
    'submission': fast_students['Assignment_Submission_Rate_%'].mean()
}

# 3. Create Tabs (Now with EDA)
tab1, tab2, tab3 = st.tabs(["🔮 Classify & Advise", "📈 Exploratory Data Analysis", "📊 View Old Records"])

with tab1:
    st.header("Enter Student Details")
    
    col1, col2 = st.columns(2)
    with col1:
        exam = st.number_input("Exam Score (0-100)", min_value=0, max_value=100, value=50)
        attendance = st.number_input("Attendance % (0-100)", min_value=0, max_value=100, value=60)
        submission = st.number_input("Assignment Submission % (0-100)", min_value=0, max_value=100, value=55)
    with col2:
        study = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        cgpa = st.number_input("Previous CGPA (0-10)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    if st.button("Predict Tier & Get Advice", type="primary"):
        extra_val = 1 if extra == "Yes" else 0
        input_data = pd.DataFrame([[exam, attendance, submission, study, cgpa, extra_val]], 
                                columns=['Exam_Score', 'Attendance_%', 'Assignment_Submission_Rate_%', 
                                         'Study_Hours_Per_Day', 'Previous_CGPA', 'Extracurricular_Activities'])
        
        prediction = model.predict(input_data)[0]
        
        # Display Prediction
        if prediction == "Fast":
            st.success(f"### Predicted Tier: {prediction} Learner 🚀\nKeep up the excellent work!")
        elif prediction == "Average":
            st.info(f"### Predicted Tier: {prediction} Learner 📚\nYou are doing well, but there is room to optimize.")
        else:
            st.warning(f"### Predicted Tier: {prediction} Learner ⚠️\nLet's look at how we can get you on track.")

        # --- THE ADVICE ENGINE ---
        if prediction in ["Slow", "Average"]:
            st.markdown("---")
            st.subheader("💡 Your Personalized Path to 'Fast Learner'")
            st.write("Here is how your current habits compare to the top-performing students in our dataset:")
            
            # Advice logic comparing user inputs to Fast Learner benchmarks
            if attendance < benchmarks['attendance']:
                st.error(f"**📉 Attendance Gap:** You are at {attendance}%. Fast learners average **{benchmarks['attendance']:.1f}%**. *Tip: Missing lectures causes compounding confusion in CS subjects. Commit to missing no more than 1 class per month.*")
            else:
                st.success(f"**✅ Attendance:** Great job! You are hitting the {benchmarks['attendance']:.1f}% average of top students.")
                
            if study < benchmarks['study']:
                st.error(f"**📉 Study Hours Gap:** You study {study} hrs/day. Fast learners average **{benchmarks['study']:.1f} hrs/day**. *Tip: Implement the Pomodoro technique (25 min focus, 5 min break) to safely increase your deep work hours.*")
            else:
                st.success(f"**✅ Study Hours:** You are putting in the necessary time alongside the top students.")
                
            if submission < benchmarks['submission']:
                st.error(f"**📉 Assignment Gap:** Your submission rate is {submission}%. Fast learners average **{benchmarks['submission']:.1f}%**. *Tip: Start assignments the day they are given to utilize the 'distributed practice' learning model.*")
            else:
                st.success(f"**✅ Assignments:** Your submission rate matches the highest performers.")

with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Deep dive into feature distributions, correlations, and class separations.")
    
    # --- ROW 1: Distribution & Density ---
    st.markdown("### 1. Data Distribution & Density")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart: Target Variable Balance
        st.write("**Class Balance (Performance Tiers)**")
        fig_pie = px.pie(df, names="Performance_Tier", hole=0.4,
                         color="Performance_Tier",
                         color_discrete_map={"Slow": "#EF553B", "Average": "#636EFA", "Fast": "#00CC96"})
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Histogram + Density (Violin) Plot: CGPA distribution
        st.write("**CGPA Distribution by Tier**")
        fig_hist = px.histogram(df, x="Previous_CGPA", color="Performance_Tier", 
                                marginal="violin", # Adds density/violin plots on the top margin
                                barmode="overlay", opacity=0.7,
                                color_discrete_map={"Slow": "#EF553B", "Average": "#636EFA", "Fast": "#00CC96"})
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: Correlations (Heatmap & Scatter) ---
    st.markdown("### 2. Feature Correlations")
    col3, col4 = st.columns(2)

    with col3:
        # Heatmap: Correlation Matrix
        st.write("**Feature Correlation Heatmap**")
        # Select only numerical columns for correlation
        num_df = df[['Exam_Score', 'Attendance_%', 'Assignment_Submission_Rate_%', 'Study_Hours_Per_Day', 'Previous_CGPA']]
        corr_matrix = num_df.corr().round(2)
        fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                             color_continuous_scale='RdBu_r', 
                             title="How strongly features link to each other")
        st.plotly_chart(fig_heat, use_container_width=True)

    with col4:
        # Scatter Plot with Trendline
        st.write("**Study Hours vs. Exam Score**")
        fig_scatter = px.scatter(df, x="Study_Hours_Per_Day", y="Exam_Score", color="Performance_Tier", 
                                 color_discrete_map={"Slow": "#EF553B", "Average": "#636EFA", "Fast": "#00CC96"},
                                 trendline="ols", # Adds ordinary least squares regression line
                                 opacity=0.7)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # --- ROW 3: Categorical & Outlier Analysis (Bar & Box) ---
    st.markdown("### 3. Feature Impact Analysis")
    col5, col6 = st.columns(2)

    with col5:
        # Grouped Bar Chart: Extracurriculars
        st.write("**Impact of Extracurriculars**")
        fig_bar = px.histogram(df, x="Performance_Tier", color="Extracurricular_Activities", 
                               barmode="group", text_auto=True,
                               category_orders={"Performance_Tier": ["Slow", "Average", "Fast"]})
        st.plotly_chart(fig_bar, use_container_width=True)

    with col6:
        # Box Plot: Attendance Outliers and Quartiles
        st.write("**Attendance Spread & Outliers**")
        fig_box = px.box(df, x="Performance_Tier", y="Attendance_%", color="Performance_Tier",
                         category_orders={"Performance_Tier": ["Slow", "Average", "Fast"]},
                         color_discrete_map={"Slow": "#EF553B", "Average": "#636EFA", "Fast": "#00CC96"})
        st.plotly_chart(fig_box, use_container_width=True)