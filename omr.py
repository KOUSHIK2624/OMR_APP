import streamlit as st
#import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageDraw
import io
import json
#import matplotlib.pyplot as plt
#from scipy import ndimage
import sqlite3
import tempfile
import os
from datetime import datetime
import warnings
#from streamlit_lottie import st_lottie
import requests
 
# Set page configuration
st.set_page_config(
    page_title="Automated OMR Evaluation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    body {
        background-color: #FAFAFA; /* Light gray background */
        font-family: 'Poppins', sans-serif !important; /* Apply Poppins globally */
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important; /* Apply Poppins to all headers */
    }
    .main-header {
        font-size: 3rem !important;
        font-family: 'Poppins', sans-serif !important; /* Apply Poppins font */
        background: linear-gradient(90deg, #FF5733, #FFC300);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color:#FF5733;
        border-bottom: 2px solid #FF5733;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #FFF3E0; /* Light orange */
        color: #FF5733; /* Bright orange text */
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #FF5733;
    }
    .success-box {
        background-color:  #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .subject-metric {
        background-color:  #FFFDE7;
        padding: 0.75rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stProgress > div > div > div > div {
        background-color: #FF5733;
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #FF5733;
        color: white;
        font-size: 0.9rem;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-item {
        background: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .student-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6366F1;
    }
    div.stButton > button:first-child {
        background-color: #FF5733;
        color: white;
        border-radius: 5px;
        height: 50px;
        width: 100%;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #FFC300;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#lottie_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_touohxv0.json")
#st_lottie(lottie_animation, height=300, key="animation")

# Initialize database
def init_db():
    # Initialize database 
    conn = sqlite3.connect('omr_results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id TEXT,
                  timestamp DATETIME,
                  sheet_version TEXT,
                  subject_scores TEXT,
                  total_score INTEGER,
                  extracted_answers TEXT,
                  correct_answers TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

# Save results to database
def save_to_db(student_id, sheet_version, subject_scores, total_score, extracted_answers, correct_answers, image_path):
    conn = sqlite3.connect('omr_results.db')
    c = conn.cursor()
    c.execute("INSERT INTO results (student_id, timestamp, sheet_version, subject_scores, total_score, extracted_answers, correct_answers, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (student_id, datetime.now(), sheet_version, json.dumps(subject_scores), total_score, json.dumps(extracted_answers), json.dumps(correct_answers), image_path))
    conn.commit()
    conn.close()
    # -------------------------------
# Define global configuration
# -------------------------------

# Answer keys for different sheet versions
answer_keys = {
    "version1": ["A", "B", "C", "D"] * 25,   # 100 questions
    "version2": ["B", "C", "D", "A"] * 25,
    "version3": ["C", "D", "A", "B"] * 25,
    "version4": ["D", "A", "B", "C"] * 25,
}

# Subjects (used for subject-wise scoring)
subjects = ["Math", "Physics", "Chemistry", "Biology", "English"]

# Default values (can be overridden in sidebar later)
sheet_version = "version1"
student_id = "S001"


# Get all results from database
def get_all_results():
    conn = sqlite3.connect('omr_results.db')
    c = conn.cursor()
    c.execute("SELECT * FROM results ORDER BY timestamp DESC")
    results = c.fetchall()
    conn.close()
    return results

# Initialize session state variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'student_id' not in st.session_state:
    st.session_state.student_id = ""

# Initialize database
init_db()

# App title and description
st.markdown('<h1 class="main-header">Automated OMR Evaluation & Scoring System</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <h3>Welcome to the Automated OMR Evaluation System</h3>
    <p>This system processes OMR sheet images captured via mobile phone cameras, extracts answers, 
    compares them with answer keys, and generates detailed score reports. Upload an OMR sheet image 
    and configure the settings to get started.</p>
</div>
""", unsafe_allow_html=True)

# Features overview
 

# 
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910765.png", width=150)
st.title("Configuration")
    
    
    
st.markdown("---")
st.subheader("Processing Options")
enable_preprocessing = st.checkbox("Enable Image Preprocessing", value=True)
sensitivity = st.slider("Bubble Detection Sensitivity", 0.1, 0.9, 0.5)
    
st.markdown("---")
if st.button("Process OMR Sheet", type="primary", use_container_width=True):
        st.session_state.processed = True
if st.button("Reset", use_container_width=True):
        st.session_state.processed = False
        st.session_state.results = None

# Collapsible "System Features" section
with st.expander("📋 System Features"):
    st.markdown("""
    - 📱 **Mobile Capture Support:** Process images captured with mobile phone cameras
    - 📊 **Multi-Version Support:** Handle 2-4 different sheet versions with unique answer keys
    - 📈 **Subject-wise Scoring:** Calculate scores for 5 subjects (0-20 each) and total score (0-100)
    - 🔍 **Bubble Detection:** Advanced computer vision techniques for accurate bubble detection
    - 🌐 **Web Application:** Easy-to-use web interface for evaluators to manage results
    - 💾 **Export Results:** Download results in JSON and CSV formats for further analysis
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["OMR Evaluation", "Results Dashboard", "Database Export"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="sub-header">Upload OMR Sheet</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image file
            image = Image.open(uploaded_file)
            st.session_state.image = np.array(image)
            
            # Display uploaded image
            st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)
            
            # Show image info
            st.info(f"Image dimensions: {image.size[0]} x {image.size[1]} pixels")
            
            # Demo of preprocessing (simulated)
            if enable_preprocessing:
                with st.expander("Preprocessing Steps"):
                    st.write("1. **Perspective Correction**: Adjusting for camera angle")
                    st.write("2. **Illumination Normalization**: Correcting lighting variations")
                    st.write("3. **Noise Reduction**: Removing artifacts and smoothing image")
                    st.write("4. **Contrast Enhancement**: Improving bubble visibility")
                    
                    # Simulate before/after images
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original Image", use_column_width=True)
                    with col2:
                        # Create a "processed" version by adding a filter
                        processed_img = image.filter(ImageFilter.EDGE_ENHANCE)
                        st.image(processed_img, caption="After Preprocessing", use_column_width=True)

    with col2:
        st.markdown('<div class="sub-header">Processing Results</div>', unsafe_allow_html=True)
        
        if st.session_state.processed and st.session_state.image is not None:
            with st.spinner("Processing OMR sheet..."):
                # Simulate processing steps
                progress_bar = st.progress(0)
                
                # Step 1: Preprocessing
                status_text = st.empty()
                status_text.text("Preprocessing image...")
                progress_bar.progress(20)
                
                # Step 2: Bubble detection
                status_text.text("Detecting bubbles...")
                progress_bar.progress(50)
                
                # Step 3: Answer extraction
                status_text.text("Extracting answers...")
                progress_bar.progress(70)
                
                # Step 4: Scoring
                status_text.text("Calculating scores...")
                progress_bar.progress(90)
                
                # Simulate results
                np.random.seed(42)  # For consistent results in demo
                subject_scores = np.random.randint(12, 21, 5)
                total_score = np.sum(subject_scores)
                
                # Generate random answers for demonstration
                extracted_answers = []
                for i in range(100):
                    if np.random.random() > 0.1:  # 90% correct answers
                        extracted_answers.append(answer_keys[sheet_version][i])
                    else:
                        # Incorrect answer
                        options = ["A", "B", "C", "D"]
                        options.remove(answer_keys[sheet_version][i])
                        extracted_answers.append(np.random.choice(options))
                
                        st.session_state.results = {
                         "subject_scores": subject_scores.tolist(),  # convert ndarray → list
                         "total_score": int(total_score),            # convert np.int64 → int
                         "extracted_answers": extracted_answers,
                         "correct_answers": answer_keys[sheet_version],
                         "subjects": subjects,
                         "sheet_version": sheet_version
  }

                
                # Save to database
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    save_to_db(
                        student_id, 
                        sheet_version, 
                        subject_scores.tolist(), 
                        int(total_score),   # ✅ FIX: cast to Python int
                        extracted_answers, 
                        answer_keys[sheet_version], 
                        tmp.name
                    )
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
            # Display results
            st.markdown('<div class="success-box"><h3>OMR Processing Complete!</h3></div>', unsafe_allow_html=True)
            
            # Overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Score", f"{st.session_state.results['total_score']}/100")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{(st.session_state.results['total_score']):.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Sheet Version", sheet_version)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Subject-wise scores
            st.subheader("Subject-wise Performance")
            subject_cols = st.columns(5)
            for i, col in enumerate(subject_cols):
                with col:
                    st.markdown(f'<div class="subject-metric">', unsafe_allow_html=True)
                    st.write(f"**{st.session_state.results['subjects'][i]}**")
                    st.metric("Score", f"{st.session_state.results['subject_scores'][i]}/20")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Score visualization
           # st.subheader("Performance Visualization")
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Subject scores bar chart
           # ax1.bar(st.session_state.results['subjects'], st.session_state.results['subject_scores'], color=['#FF5733', '#FFC300', '#FF8C00', '#FFB74D', '#FFD54F'])
           # ax1.set_ylabel('Score')
           # ax1.set_title('Subject-wise Scores')
           # ax1.set_ylim(0, 20)
           # ax1.tick_params(axis='x', rotation=45)
            
            # Overall score pie chart
           # correct = st.session_state.results['total_score']
            #incorrect = 100 - correct
           # ax2.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
           # ax2.set_title('Overall Performance')
            
           # st.pyplot(fig)
            
            # Answer comparison
            st.subheader("Answer Comparison")
            comparison_data = []
            for i in range(min(20, len(st.session_state.results['extracted_answers']))):  # Show first 20
                is_correct = st.session_state.results['extracted_answers'][i] == st.session_state.results['correct_answers'][i]
                comparison_data.append({
                    "Question": i+1,
                    "Extracted": st.session_state.results['extracted_answers'][i],
                    "Correct": st.session_state.results['correct_answers'][i],
                    "Status": "Correct" if is_correct else "Incorrect"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Export options
            st.subheader("Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps(st.session_state.results, indent=2)
                st.download_button(
                    label="Download JSON Results",
                    data=json_data,
                    file_name="omr_results.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV export
                csv_data = pd.DataFrame({
                    'Subject': st.session_state.results['subjects'],
                    'Score': st.session_state.results['subject_scores']
                })
                csv = csv_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV Report",
                    data=csv,
                    file_name="omr_report.csv",
                    mime="text/csv"
                )
        
        elif not st.session_state.processed:
            st.info("Upload an OMR sheet image and click 'Process OMR Sheet' to begin evaluation.")
        else:
            st.warning("Please upload an OMR sheet image to process.")

with tab2:
    st.markdown('<div class="sub-header">Results Dashboard</div>', unsafe_allow_html=True)
    
    # Get all results from database
    results = get_all_results()
    
    if results:
        st.subheader("Recent Evaluations")
        
        for result in results[:5]:  # Show only the 5 most recent results
            with st.expander(f"Student ID: {result[1]} | Score: {result[5]}/100 | Date: {result[2]}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Sheet Version:** {result[3]}")
                    st.write(f"**Total Score:** {result[5]}/100")
                    
                    # Display subject scores
                    subject_scores = json.loads(result[4])
                    st.write("**Subject Scores:**")
                    for i, score in enumerate(subject_scores):
                        st.write(f"- {subjects[i] if i < len(subjects) else f'Subject {i+1}'}: {score}/20")
                
                with col2:
                    # Show answer accuracy
                    extracted = json.loads(result[6])
                    correct = json.loads(result[7])
                    accuracy = sum(1 for i in range(len(extracted)) if extracted[i] == correct[i]) / len(extracted) * 100
                    st.write(f"**Accuracy:** {accuracy:.2f}%")
                    
                    # Show image if available
                    if result[8] and os.path.exists(result[8]):
                        try:
                            img = Image.open(result[8])
                            st.image(img, caption="Processed OMR Sheet", width=200)
                        except:
                            st.write("Image not available")
        
        st.subheader("Performance Statistics")
        
        # ✅ FIX: Handle binary data conversion properly
        total_scores = []
        for r in results:
            if r[5] is not None:
                if isinstance(r[5], bytes):
                    # Convert binary data to integer
                    try:
                        total_scores.append(int.from_bytes(r[5], byteorder='little'))
                    except:
                        total_scores.append(0)
                else:
                    total_scores.append(int(r[5]))
        
        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Evaluations", len(results))
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        with col3:
            st.metric("Best Score", f"{max(total_scores) if total_scores else 0}/100")
        
        # Display score distribution
       # st.write("**Score Distribution**")
       # fig, ax = plt.subplots(figsize=(10, 4))
       # ax.hist(total_scores, bins=10, color='skyblue', edgecolor='black')
       # ax.set_xlabel('Score')
       # ax.set_ylabel('Frequency')
       # ax.set_title('Distribution of Scores')
       # st.pyplot(fig)
        
    else:
        st.info("No evaluation results yet. Process some OMR sheets to see data here.")

with tab3:
    st.markdown('<div class="sub-header">Database Export</div>', unsafe_allow_html=True)
    
    # Get all results from database
    results = get_all_results()
    
    if results:
        # Convert to DataFrame
        data = []
        for result in results:
            subject_scores = json.loads(result[4])
            # Handle binary data conversion for total_score
            if isinstance(result[5], bytes):
                total_score = int.from_bytes(result[5], byteorder='little')
            else:
                total_score = result[5]
                
                data.append({
                "Student ID": result[1],
                "Timestamp": result[2],
                "Sheet Version": result[3],
                "Subject 1 Score": subject_scores[0] if len(subject_scores) > 0 else 0,
                "Subject 2 Score": subject_scores[1] if len(subject_scores) > 1 else 0,
                "Subject 3 Score": subject_scores[2] if len(subject_scores) > 2 else 0,
                "Subject 4 Score": subject_scores[3] if len(subject_scores) > 3 else 0,
                "Subject 5 Score": subject_scores[4] if len(subject_scores) > 4 else 0,
                "Total Score": total_score
            })
df = pd.DataFrame(data)

# Display data
st.dataframe(df, use_container_width=True)

# Export options
st.subheader("Export All Results")

col1, col2 = st.columns(2)

with col1:
    # CSV export
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV (All Results)",
        data=csv,
        file_name="all_omr_results.csv",
        mime="text/csv"
    )

with col2:
    # JSON export
    json_data = df.to_json(orient='records', indent=2)
    st.download_button(
        label="Download JSON (All Results)",
        data=json_data,
        file_name="all_omr_results.json",
        mime="application/json"
    )
else:
     st.info("NO evaluation results yet. Process some OMR sheets to export data.")
