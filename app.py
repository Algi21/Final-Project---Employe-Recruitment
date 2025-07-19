import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Konfigurasi halaman
st.set_page_config(
    page_title="Employee Eligibility Prediction",
    page_icon="Logo_Tim.jpeg",
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

# Load and encode logo
try:
    import base64
    if 'logo_base64' not in st.session_state:
        with open("Logo_Tim.jpeg", "rb") as img_file:
            st.session_state.logo_base64 = base64.b64encode(img_file.read()).decode()
        # Re-run to update the header with logo
        st.rerun()
except FileNotFoundError:
    st.warning("Logo file 'Logo_Tim.jpeg' tidak ditemukan. Pastikan file logo tersedia di directory yang sama dengan app.py")

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
        <img src="data:image/jpeg;base64,{}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
        <div>
            <h1 style="margin: 0;">Employee Eligibility Prediction System</h1>
            <p style="margin: 0;">Prediksi Kelayakan Kandidat Karyawan dengan Machine Learning</p>
        </div>
    </div>
</div>
""".format(st.session_state.get('logo_base64', '')), unsafe_allow_html=True)

# Load model dan preprocessing components
@st.cache_resource
def load_model_and_preprocessors():
    try:
        # Load model
        model = joblib.load('LGBM_tuned.pkl')
        
        # Load preprocessing components
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        
        # Debug info tentang scaler
        scaler_info = {}
        if hasattr(scaler, 'feature_names_in_'):
            scaler_info['features'] = list(scaler.feature_names_in_)
            scaler_info['n_features'] = len(scaler.feature_names_in_)
        else:
            scaler_info['features'] = "Feature names not available"
            scaler_info['n_features'] = "Unknown"
        
        if hasattr(scaler, 'scale_'):
            scaler_info['n_features_fit'] = len(scaler.scale_)
        
        st.success("✅ Model dan preprocessing components berhasil dimuat!")
        
        # Show scaler info in expander
        with st.expander("🔍 Informasi Scaler"):
            st.write("**Scaler Information:**")
            st.json(scaler_info)
        
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"❌ File tidak ditemukan: {str(e)}")
        st.info("Pastikan file berikut tersedia di directory yang sama dengan app.py:")
        st.info("- LGBM_tuned.pkl")
        st.info("- scaler.pkl") 
        st.info("- encoder.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading files: {str(e)}")
        return None, None, None

# Fungsi untuk prediksi dengan preprocessing
def predict_eligibility(features):
    model, scaler, encoder = load_model_and_preprocessors()
    if model is not None and scaler is not None:
        try:
            # Convert to DataFrame untuk memudahkan preprocessing
            feature_df = pd.DataFrame([features], columns=CORRECT_FEATURE_ORDER)
            
            # Debug: Cek fitur apa yang diharapkan oleh scaler
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
                st.info(f"🔍 Fitur yang diharapkan oleh scaler: {list(expected_features)}")
                
                # Gunakan hanya fitur yang diharapkan oleh scaler
                available_features = [f for f in expected_features if f in feature_df.columns]
                missing_features = [f for f in expected_features if f not in feature_df.columns]
                
                if missing_features:
                    st.warning(f"⚠️ Fitur yang hilang untuk scaler: {missing_features}")
                
                # Apply scaler hanya pada fitur yang tersedia dan diharapkan
                feature_df_scaled = feature_df.copy()
                if available_features:
                    feature_df_scaled[available_features] = scaler.transform(feature_df[available_features])
                
            else:
                # Jika scaler tidak memiliki feature_names_in_, gunakan pendekatan lama
                # Coba dengan semua fitur terlebih dahulu
                try:
                    feature_df_scaled = feature_df.copy()
                    feature_df_scaled[CORRECT_FEATURE_ORDER] = scaler.transform(feature_df[CORRECT_FEATURE_ORDER])
                except Exception as scale_error:
                    st.error(f"❌ Error dengan scaler: {str(scale_error)}")
                    # Fallback: gunakan fitur tanpa scaling
                    st.warning("⚠️ Menggunakan fitur tanpa preprocessing (ini mungkin tidak akurat)")
                    feature_df_scaled = feature_df.copy()
            
            # Convert kembali ke array dengan urutan yang benar
            features_processed = feature_df_scaled[CORRECT_FEATURE_ORDER].values
            
            # Prediksi
            prediction = model.predict(features_processed)
            probability = model.predict_proba(features_processed)
            
            return prediction[0], probability[0]
            
        except Exception as e:
            st.error(f"❌ Error dalam preprocessing atau prediksi: {str(e)}")
            return None, None
    return None, None

# Fungsi untuk batch prediction dengan preprocessing
def batch_predict_eligibility(df):
    model, scaler, encoder = load_model_and_preprocessors()
    if model is not None and scaler is not None:
        try:
            # Prepare features dengan urutan yang benar
            features_df = df[CORRECT_FEATURE_ORDER].copy()
            
            # Debug: Cek fitur apa yang diharapkan oleh scaler
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
                st.info(f"🔍 Fitur yang diharapkan oleh scaler: {list(expected_features)}")
                
                # Gunakan hanya fitur yang diharapkan oleh scaler
                available_features = [f for f in expected_features if f in features_df.columns]
                missing_features = [f for f in expected_features if f not in features_df.columns]
                
                if missing_features:
                    st.warning(f"⚠️ Fitur yang hilang untuk scaler: {missing_features}")
                
                # Apply scaler hanya pada fitur yang tersedia dan diharapkan
                if available_features:
                    features_df[available_features] = scaler.transform(features_df[available_features])
                
            else:
                # Jika scaler tidak memiliki feature_names_in_, coba dengan semua fitur
                try:
                    features_df[CORRECT_FEATURE_ORDER] = scaler.transform(features_df[CORRECT_FEATURE_ORDER])
                except Exception as scale_error:
                    st.error(f"❌ Error dengan scaler: {str(scale_error)}")
                    st.warning("⚠️ Menggunakan fitur tanpa preprocessing (ini mungkin tidak akurat)")
            
            # Convert ke array
            features = features_df.values
            
            # Prediksi
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            return predictions, probabilities
            
        except Exception as e:
            st.error(f"❌ Error dalam batch prediction: {str(e)}")
            return None, None
    return None, None

# Fungsi untuk menghitung total skills
def calculate_total_skills(skills):
    return sum(1 for skill in skills.values() if skill)

# Fungsi untuk menghitung edu_exp_score (multiplication formula)
def calculate_edu_exp_score(education_level, years_experience):
    # Formula: education_level * years_experience
    return education_level * years_experience

# Urutan fitur yang BENAR sesuai training data
CORRECT_FEATURE_ORDER = [
    'education_level',       # 0
    'years_experience',      # 1
    'num_relevant_skills',   # 2
    'internal_referral',     # 3
    'interview_score',       # 4
    'technical_test_score',  # 5
    'Cloud_Computing',       # 6
    'Communication',         # 7
    'Data_Visualization',    # 8
    'Excel',                 # 9
    'Leadership',            # 10
    'Machine_Learning',      # 11
    'Python',                # 12
    'SQL',                   # 13
    'total_skills',          # 14
    'avg_test_score',        # 15
    'edu_exp_score'          # 16
]

# Tabs
tab1, tab2 = st.tabs(["🔍 Prediksi Individual", "📊 Prediksi Batch & Analisis"])

with tab1:
    st.header("Prediksi Kelayakan Kandidat Individual")
    
    # Load components untuk menampilkan status
    model, scaler, encoder = load_model_and_preprocessors()
    
    if model is not None and scaler is not None:
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
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Skor hasil wawancara (1-10)"
            )
            
            # Technical Test Score
            technical_test_score = st.slider(
                "💻 Technical Test Score",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.1,
                help="Skor tes teknis (1-10)"
            )
            
            # Average Test Score akan dihitung otomatis
            st.info("📝 Average Test Score akan dihitung otomatis dari Interview Score dan Technical Test Score")
        
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
        
        # Calculate average test score
        avg_test_score = (interview_score + technical_test_score) / 2
        
        # Calculate education-experience score
        edu_exp_score = calculate_edu_exp_score(education_options[education_level], years_exp_processed)
        
        # Show calculated values
        st.subheader("🔢 Nilai yang Dihitung Otomatis")
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        with calc_col1:
            st.metric("Total Skills", total_skills)
        with calc_col2:
            st.metric("Avg Test Score", f"{avg_test_score:.1f}")
        with calc_col3:
            st.metric("Edu-Exp Score", edu_exp_score)
        
        # Prediction button
        if st.button("🚀 Prediksi Kelayakan", type="primary"):
            # Prepare features array dengan urutan yang BENAR
            features = [
                education_options[education_level],  # 0: education_level
                years_exp_processed,                 # 1: years_experience
                num_relevant_skills,                 # 2: num_relevant_skills
                1 if internal_referral else 0,      # 3: internal_referral
                interview_score,                     # 4: interview_score
                technical_test_score,                # 5: technical_test_score
                1 if cloud else 0,                  # 6: Cloud_Computing
                1 if comm else 0,                   # 7: Communication
                1 if viz else 0,                    # 8: Data_Visualization
                1 if excel else 0,                  # 9: Excel
                1 if lead else 0,                   # 10: Leadership
                1 if ml else 0,                     # 11: Machine_Learning
                1 if python else 0,                 # 12: Python
                1 if sql else 0,                    # 13: SQL
                total_skills,                       # 14: total_skills
                avg_test_score,                     # 15: avg_test_score
                edu_exp_score                       # 16: edu_exp_score
            ]
            
            # Debug: Show feature values (raw)
            st.subheader("🔍 Debug - Raw Feature Values")
            debug_df = pd.DataFrame({
                'Feature': CORRECT_FEATURE_ORDER,
                'Raw Value': features
            })
            st.dataframe(debug_df, hide_index=True)
            
            # Option to run without preprocessing
            use_preprocessing = st.checkbox("🔧 Gunakan Preprocessing (Recommended)", value=True)
            
            if use_preprocessing:
                # Make prediction with preprocessing
                prediction, probability = predict_eligibility(features)
            else:
                # Make prediction without preprocessing (fallback)
                st.warning("⚠️ Menjalankan prediksi tanpa preprocessing - hasil mungkin tidak akurat")
                model, _, _ = load_model_and_preprocessors()
                if model is not None:
                    try:
                        prediction = model.predict([features])[0]
                        probability = model.predict_proba([features])[0]
                    except Exception as e:
                        st.error(f"❌ Error dalam prediksi: {str(e)}")
                        prediction, probability = None, None
                else:
                    prediction, probability = None, None
            
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
                
                # NEW LAYOUT: Chart and Summary side by side
                chart_col, summary_col = st.columns([1, 1])
                
                with chart_col:
                    st.subheader("📊 Probability Distribution")
                    # Smaller probability chart
                    fig, ax = plt.subplots(figsize=(6, 4))
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
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with summary_col:
                    # Feature summary
                    st.subheader("📊 Ringkasan Fitur")
                    summary_data = {
                        'Feature': ['Education Level', 'Years Experience', 'Interview Score', 
                                   'Technical Score', 'Total Skills', 'Avg Test Score', 'Edu-Exp Score'],
                        'Value': [education_level, years_experience, interview_score,
                                 technical_test_score, total_skills, avg_test_score, edu_exp_score]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    else:
        st.error("❌ Model atau preprocessing components tidak dapat dimuat. Pastikan semua file tersedia.")

with tab2:
    st.header("Prediksi Batch")
    
    # Load components untuk menampilkan status
    model, scaler, encoder = load_model_and_preprocessors()
    
    if model is not None and scaler is not None:
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
                
                # Check columns - menggunakan urutan yang BENAR
                expected_columns = CORRECT_FEATURE_ORDER
                
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    st.warning(f"⚠️ Kolom yang hilang: {', '.join(missing_columns)}")
                    st.info("Pastikan CSV memiliki semua kolom yang diperlukan untuk prediksi")
                    
                    # Show expected column order
                    st.subheader("📋 Urutan Kolom yang Diharapkan")
                    expected_df = pd.DataFrame({
                        'Index': range(len(expected_columns)),
                        'Column Name': expected_columns
                    })
                    st.dataframe(expected_df, hide_index=True)
                else:
                    # Make predictions
                    if st.button("🔮 Jalankan Prediksi Batch", type="primary"):
                        # Debug: Show first few rows of raw features
                        st.subheader("🔍 Debug - Raw Feature Preview")
                        debug_preview = df[expected_columns].head()
                        st.dataframe(debug_preview, use_container_width=True)
                        
                        # Predict with preprocessing
                        predictions, probabilities = batch_predict_eligibility(df)
                        
                        if predictions is not None:
                            # Add results to dataframe
                            df['prediction'] = predictions
                            df['prediction_label'] = ['Eligible' if p == 1 else 'Not Eligible' for p in predictions]
                            df['probability_eligible'] = probabilities[:, 1]
                            df['confidence'] = np.max(probabilities, axis=1) * 100
                            
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
                            
                            # Visualization Section
                            st.subheader("📈 Visualisasi Data")
                            
                            # Create three columns for visualizations
                            viz_col1, viz_col2, viz_col3 = st.columns(3)
                            
                            with viz_col1:
                                # Pie chart for predictions (tetap ada)
                                fig1, ax1 = plt.subplots(figsize=(6, 6))
                                labels = ['Eligible', 'Not Eligible']
                                sizes = [eligible_count, not_eligible_count]
                                colors = ['#51cf66', '#ff6b6b']
                                
                                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                                ax1.set_title('Distribution of Predictions')
                                st.pyplot(fig1)
                            
                            with viz_col2:
                                # Stacked bar chart: Education Level vs Prediction
                                if 'education_level' in df.columns:
                                    # Create education labels
                                    education_labels = {0: 'Bachelor', 1: 'Master', 2: 'PhD'}
                                    df['education_label'] = df['education_level'].map(education_labels)
                                    
                                    # Create crosstab for education vs prediction
                                    edu_pred_crosstab = pd.crosstab(df['education_label'], df['prediction_label'])
                                    
                                    # Calculate percentages
                                    edu_pred_pct = edu_pred_crosstab.div(edu_pred_crosstab.sum(axis=1), axis=0) * 100
                                    
                                    fig2, ax2 = plt.subplots(figsize=(8, 6))
                                    edu_pred_pct.plot(kind='bar', stacked=True, ax=ax2, 
                                                     color=['#51cf66', '#ff6b6b'], width=0.7)
                                    ax2.set_title('Education Level vs Prediction (%)')
                                    ax2.set_xlabel('Education Level')
                                    ax2.set_ylabel('Percentage (%)')
                                    ax2.legend(title='Prediction')
                                    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                                    
                                    # Add percentage labels on bars
                                    for container in ax2.containers:
                                        ax2.bar_label(container, fmt='%.1f%%', label_type='center')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig2)
                            
                            with viz_col3:
                                # Stacked bar chart: Internal Referral vs Prediction
                                if 'internal_referral' in df.columns:
                                    # Create referral labels
                                    df['referral_label'] = df['internal_referral'].map({0: 'No Referral', 1: 'With Referral'})
                                    
                                    # Create crosstab for referral vs prediction
                                    ref_pred_crosstab = pd.crosstab(df['referral_label'], df['prediction_label'])
                                    
                                    # Calculate percentages
                                    ref_pred_pct = ref_pred_crosstab.div(ref_pred_crosstab.sum(axis=1), axis=0) * 100
                                    
                                    fig3, ax3 = plt.subplots(figsize=(8, 6))
                                    ref_pred_pct.plot(kind='bar', stacked=True, ax=ax3, 
                                                     color=['#51cf66', '#ff6b6b'], width=0.7)
                                    ax3.set_title('Internal Referral vs Prediction (%)')
                                    ax3.set_xlabel('Internal Referral')
                                    ax3.set_ylabel('Percentage (%)')
                                    ax3.legend(title='Prediction')
                                    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
                                    
                                    # Add percentage labels on bars
                                    for container in ax3.containers:
                                        ax3.bar_label(container, fmt='%.1f%%', label_type='center')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig3)

                            # KDE Distribution Plots
                            st.subheader("📊 Feature Distribution by Eligibility")
                            
                            # Define numeric columns for KDE plots
                            nums = ['years_experience', 'num_relevant_skills', 'interview_score',
                                    'technical_test_score', 'avg_test_score']
                            
                            # Check which columns exist in the dataframe
                            available_nums = [col for col in nums if col in df.columns]
                            
                            if available_nums:
                                cols = 3
                                rows = math.ceil(len(available_nums) / cols)
                                
                                fig4, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
                                
                                # Handle single row case
                                if rows == 1:
                                    axes = axes.reshape(1, -1)
                                elif rows == 1 and cols == 1:
                                    axes = np.array([axes])
                                
                                # Flatten axes for easier indexing
                                axes_flat = axes.flatten()
                                
                                for i, col in enumerate(available_nums):
                                    ax = axes_flat[i]
                                    
                                    # Create KDE plot
                                    sns.kdeplot(
                                        data=df,
                                        x=col,
                                        hue='prediction_label',
                                        fill=True,
                                        common_norm=False,
                                        alpha=0.4,
                                        linewidth=1.5,
                                        palette={'Eligible': '#008080', 'Not Eligible': '#55acee'},
                                        ax=ax
                                    )
                                    
                                    ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
                                    ax.set_xlabel(col.replace("_", " ").title())
                                    ax.set_ylabel('Density')
                                
                                # Hide unused subplots
                                for j in range(len(available_nums), len(axes_flat)):
                                    axes_flat[j].set_visible(False)
                                
                                plt.tight_layout()
                                st.pyplot(fig4)
                                
                                # Add interpretation text
                                st.info("""
                                💡 **Interpretasi Grafik:**
                                - Distribusi hijau (Eligible) menunjukkan pola nilai untuk kandidat yang layak
                                - Distribusi biru (Not Eligible) menunjukkan pola nilai untuk kandidat yang tidak layak
                                - Perbedaan yang jelas antara kedua distribusi menunjukkan fitur yang diskriminatif
                                """)
                            else:
                                st.warning("⚠️ Kolom numerik yang diperlukan tidak ditemukan dalam data")
                                st.info(f"Kolom yang dicari: {', '.join(nums)}")
                                st.info(f"Kolom yang tersedia: {', '.join(df.columns.tolist())}")
                            
       
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
    
    else:
        st.error("❌ Model atau preprocessing components tidak dapat dimuat. Pastikan semua file tersedia.")
# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Employee Eligibility Prediction System | Built with Streamlit & Machine Learning</p>
        <p><small>🔧 Updated: Preprocessing pipeline (scaler & encoder) integrated</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
