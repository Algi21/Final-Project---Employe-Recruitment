import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(
    page_title="Employee Eligibility Prediction",
    page_icon="Logo_Tim.JPEG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>👨‍💼 Employee Eligibility Prediction System</h1>
    <p>Prediksi Kelayakan Kandidat Karyawan dengan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Load model (pastikan model sudah tersedia)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('LGBM_tuned.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'LGBM_tuned.pkl' tidak ditemukan. Pastikan file model sudah tersedia.")
        return None

# Fungsi untuk prediksi
def predict_eligibility(features):
    model = load_model()
    if model is not None:
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        return prediction[0], probability[0]
    return None, None

# Fungsi untuk menghitung total skills
def calculate_total_skills(skills):
    return sum(1 for skill in skills.values() if skill)

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediksi Individual", "📊 Prediksi Batch & Analisis"])

with tab1:
    st.header("Prediksi Kelayakan Kandidat Individual")
    
    # Layout dengan kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Data Pribadi & Pengalaman")
        
        # Education Level
        education_options = {
            "Bachelor Degree": 0,
            "Master Degree": 1,
            "PhD": 2
        }
        education_level = st.selectbox(
            "🎓 Education Level",
            options=list(education_options.keys()),
            help="Pilih tingkat pendidikan terakhir"
        )
        
        # Years Experience
        years_exp_options = list(range(0, 21))
        years_exp_options.append(">20")
        years_experience = st.selectbox(
            "💼 Years Experience",
            options=years_exp_options,
            help="Jumlah tahun pengalaman kerja"
        )
        
        # Relevant Skills Count
        num_relevant_skills = st.selectbox(
            "🔧 Number of Relevant Skills",
            options=list(range(0, 9)),
            help="Jumlah keahlian yang relevan (0-8)"
        )
        
        # Internal Referral
        internal_referral = st.checkbox(
            "🤝 Internal Referral",
            help="Apakah kandidat memiliki referensi internal?"
        )
        
    with col2:
        st.subheader("📊 Skor Penilaian")
        
        # Interview Score
        interview_score = st.slider(
            "🎤 Interview Score",
            min_value=1,
            max_value=10,
            value=5,
            help="Skor hasil wawancara (1-10)"
        )
        
        # Technical Test Score
        technical_test_score = st.slider(
            "💻 Technical Test Score",
            min_value=1,
            max_value=10,
            value=5,
            help="Skor tes teknis (1-10)"
        )
        
        # Average Test Score
        avg_test_score = st.slider(
            "📝 Average Test Score",
            min_value=1,
            max_value=10,
            value=5,
            help="Rata-rata skor tes (1-10)"
        )
        
        # Education Experience Score
        edu_exp_score = st.number_input(
            "🎯 Education-Experience Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
            help="Skor gabungan pendidikan dan pengalaman"
        )
    
    # Skills Section
    st.subheader("🛠️ Technical Skills")
    skills_col1, skills_col2 = st.columns(2)
    
    with skills_col1:
        cloud = st.checkbox("☁️ Cloud Computing")
        comm = st.checkbox("💬 Communication")
        viz = st.checkbox("📊 Data Visualization")
        excel = st.checkbox("📊 Excel")
    
    with skills_col2:
        lead = st.checkbox("👑 Leadership")
        ml = st.checkbox("🤖 Machine Learning")
        python = st.checkbox("🐍 Python")
        sql = st.checkbox("🗄️ SQL")
    
    # Collect skills
    skills = {
        'cloud': cloud,
        'communication': comm,
        'data_visualization': viz,
        'excel': excel,
        'leadership': lead,
        'machine_learning': ml,
        'python': python,
        'sql': sql
    }
    
    # Calculate derived features
    total_skills = calculate_total_skills(skills)
    
    # Process years experience
    years_exp_processed = years_experience if years_experience != ">20" else 21
    
    # Prediction button
    if st.button("🚀 Prediksi Kelayakan", type="primary"):
        # Prepare features array
        features = [
            education_options[education_level],  # education_level
            years_exp_processed,  # years_experience
            num_relevant_skills,  # num_relevant_skills
            1 if internal_referral else 0,  # internal_referral
            interview_score,  # interview_score
            technical_test_score,  # technical_test_score
            avg_test_score,  # avg_test_score
            edu_exp_score,  # edu_exp_score
            total_skills,  # total_skills
            1 if cloud else 0,  # Cloud_Computing
            1 if comm else 0,  # Communication
            1 if viz else 0,  # Data_Visualization
            1 if excel else 0,  # Excel
            1 if lead else 0,  # Leadership
            1 if ml else 0,  # Machine_Learning
            1 if python else 0,  # Python
            1 if sql else 0  # SQL
        ]
        
        # Make prediction
        prediction, probability = predict_eligibility(features)
        
        if prediction is not None:
            # Display results
            st.subheader("📋 Hasil Prediksi")
            
            # Result columns
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                if prediction == 1:
                    st.success("✅ **ELIGIBLE** - Kandidat layak diterima")
                else:
                    st.error("❌ **NOT ELIGIBLE** - Kandidat tidak layak diterima")
            
            with result_col2:
                confidence = max(probability) * 100
                st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Probability chart
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Not Eligible', 'Eligible']
            probs = [probability[0], probability[1]]
            colors = ['#ff6b6b', '#51cf66']
            
            bars = ax.bar(categories, probs, color=colors)
            ax.set_ylabel('Probability')
            ax.set_title('Probability Distribution')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Feature summary
            st.subheader("📊 Ringkasan Fitur")
            summary_data = {
                'Feature': ['Education Level', 'Years Experience', 'Interview Score', 
                           'Technical Score', 'Total Skills'],
                'Value': [education_level, years_experience, interview_score,
                         technical_test_score, total_skills]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True)

with tab2:
    st.header("Prediksi Batch & Analisis Data")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload CSV File",
        type=['csv'],
        help="Upload file CSV dengan kolom yang sesuai dengan fitur model"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil diupload! Data shape: {df.shape}")
            
            # Display first few rows
            st.subheader("👀 Preview Data")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check columns
            expected_columns = [
                'education_level', 'years_experience', 'num_relevant_skills',
                'internal_referral', 'interview_score', 'technical_test_score',
                'avg_test_score', 'edu_exp_score', 'total_skills',
                'Cloud_Computing', 'Communication', 'Data_Visualization', 'Excel',
                'Leadership', 'Machine_Learning', 'Python', 'SQL'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"⚠️ Kolom yang hilang: {', '.join(missing_columns)}")
                st.info("Pastikan CSV memiliki semua kolom yang diperlukan untuk prediksi")
            else:
                # Make predictions
                if st.button("🔮 Jalankan Prediksi Batch", type="primary"):
                    model = load_model()
                    if model is not None:
                        # Prepare features
                        features = df[expected_columns].values
                        
                        # Predict
                        predictions = model.predict(features)
                        probabilities = model.predict_proba(features)
                        
                        # Add results to dataframe
                        df['prediction'] = predictions
                        df['prediction_label'] = ['Eligible' if p == 1 else 'Not Eligible' for p in predictions]
                        df['probability_eligible'] = probabilities[:, 1]
                        df['confidence'] = np.max(probabilities, axis=1)
                        
                        # Display results
                        st.subheader("📊 Hasil Prediksi")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_candidates = len(df)
                            st.metric("Total Candidates", total_candidates)
                        
                        with col2:
                            eligible_count = sum(predictions)
                            st.metric("Eligible", eligible_count)
                        
                        with col3:
                            not_eligible_count = total_candidates - eligible_count
                            st.metric("Not Eligible", not_eligible_count)
                        
                        with col4:
                            avg_confidence = df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                        
                        # Visualization
                        fig_col1, fig_col2 = st.columns(2)
                        
                        with fig_col1:
                            # Pie chart for predictions
                            fig1, ax1 = plt.subplots(figsize=(6, 6))
                            labels = ['Eligible', 'Not Eligible']
                            sizes = [eligible_count, not_eligible_count]
                            colors = ['#51cf66', '#ff6b6b']
                            
                            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                            ax1.set_title('Distribution of Predictions')
                            st.pyplot(fig1)
                        
                        with fig_col2:
                            # Histogram of confidence scores
                            fig2, ax2 = plt.subplots(figsize=(6, 6))
                            ax2.hist(df['confidence'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                            ax2.set_xlabel('Confidence Score')
                            ax2.set_ylabel('Count')
                            ax2.set_title('Distribution of Confidence Scores')
                            st.pyplot(fig2)
                        
                        # Feature analysis
                        st.subheader("📈 Analisis Fitur")
                        
                        # Feature importance by education level
                        if 'education_level' in df.columns:
                            fig3, ax3 = plt.subplots(figsize=(10, 6))
                            education_labels = {0: 'Bachelor', 1: 'Master', 2: 'PhD'}
                            df['education_label'] = df['education_level'].map(education_labels)
                            
                            # Box plot using seaborn
                            sns.boxplot(data=df, x='education_label', y='probability_eligible', ax=ax3)
                            ax3.set_title('Probability by Education Level')
                            ax3.set_xlabel('Education Level')
                            ax3.set_ylabel('Probability Eligible')
                            st.pyplot(fig3)
                        
                        # Correlation with prediction
                        numeric_columns = df.select_dtypes(include=[np.number]).columns
                        corr_with_pred = df[numeric_columns].corrwith(df['prediction']).sort_values(ascending=False)
                        
                        # Remove NaN values
                        corr_with_pred = corr_with_pred.dropna()
                        
                        fig4, ax4 = plt.subplots(figsize=(10, 8))
                        y_pos = np.arange(len(corr_with_pred))
                        
                        bars = ax4.barh(y_pos, corr_with_pred.values)
                        ax4.set_yticks(y_pos)
                        ax4.set_yticklabels(corr_with_pred.index)
                        ax4.set_xlabel('Correlation with Prediction')
                        ax4.set_title('Feature Correlation with Prediction')
                        
                        # Color bars based on correlation value
                        for i, (bar, corr_val) in enumerate(zip(bars, corr_with_pred.values)):
                            if corr_val > 0:
                                bar.set_color('green')
                            else:
                                bar.set_color('red')
                        
                        plt.tight_layout()
                        st.pyplot(fig4)
                        
                        # Display full results
                        st.subheader("📋 Hasil Lengkap")
                        result_df = df[['prediction_label', 'probability_eligible', 'confidence'] + expected_columns]
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results as CSV",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Pastikan file CSV memiliki format yang benar")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Employee Eligibility Prediction System | Built with Streamlit & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
