import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from components import InterviewAssistant  # Assuming this handles audio/text processing

# Configuration
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-specific-password"
GITHUB_LINK = "https://github.com/yourusername/project"
TEAM_WEBSITE = "https://yourteam.com"

# Set up directories
Path("temp_audio").mkdir(exist_ok=True)

def generate_mock_job_trends():
    """Generate mock job trends data for visualization"""
    dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='M')
    jobs = {
        'Data Scientist': np.random.normal(100, 10, len(dates)) * 1.5,
        'AI Engineer': np.random.normal(80, 15, len(dates)) * 2,
        'ML Engineer': np.random.normal(90, 12, len(dates)) * 1.8,
        'Data Engineer': np.random.normal(85, 8, len(dates)) * 1.6,
        'Cloud Engineer': np.random.normal(95, 11, len(dates)) * 1.7
    }
    
    fig = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    for (job, values), color in zip(jobs.items(), colors):
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=job,
            line=dict(color=color, width=2),
            mode='lines'
        ))
    
    fig.update_layout(
        title="Job Market Trends (2019-2024)",
        xaxis_title="Year",
        yaxis_title="Job Postings (Normalized)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

def send_feedback_email(email, name, feedback):
    """Send feedback to user's email"""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = email
    msg['Subject'] = f"Your Interview Feedback - {datetime.now().strftime('%Y-%m-%d')}"
    
    body = f"""
    Dear {name},
    Thank you for completing your interview practice session. Here's your feedback:
    {feedback}
    Best regards,
    Interview Assistant Team
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

class InterviewApp:
    def __init__(self):
        self.assistant = InterviewAssistant(
            questions_path='non_technical_interview_questions.csv',
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        self.current_question = 0
        self.questions = []
        self.answers = []
        self.user_info = {}

    def run(self):
        st.set_page_config(page_title="AI Interview Assistant", layout="centered")
        
        # Navigation bar
        st.sidebar.title("Navigation")
        st.sidebar.write(f"[GitHub Project]({GITHUB_LINK})")
        st.sidebar.write(f"[About Our Team]({TEAM_WEBSITE})")

        # Welcome Page
        if "page" not in st.session_state:
            st.session_state.page = "welcome"

        if st.session_state.page == "welcome":
            self.welcome_page()
        elif st.session_state.page == "user_info":
            self.user_info_page()
        elif st.session_state.page == "interview":
            self.interview_page()
        elif st.session_state.page == "feedback":
            self.feedback_page()

    def welcome_page(self):
        st.title("AI Interview Assistant")
        st.write("""
            Welcome! This app helps you practice job interviews with AI-generated questions 
            and provides personalized feedback.
        """)
        st.plotly_chart(generate_mock_job_trends())
        
        if st.button("Start Interview"):
            st.session_state.page = "user_info"

    def user_info_page(self):
        st.title("User Information")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        job_title = st.text_input("Desired Job Title")
        job_desc = st.text_area("Job Description")
        
        if st.button("Start Interview"):
            self.user_info = {
                "name": name,
                "email": email,
                "job_title": job_title,
                "job_desc": job_desc
            }
            
            st.session_state["questions"] = self.assistant.get_similar_questions(job_title, job_desc)
            st.session_state["audio_paths"] = []
            
            # Convert questions to audio and display "Generating Questions"
            with st.spinner("Generating Questions..."):
                for i, question in enumerate(st.session_state["questions"]):
                    audio_path = f"temp_audio/question_{i}.mp3"
                    self.assistant.text_to_speech(question["Question"], audio_path)
                    st.session_state["audio_paths"].append(audio_path)
                    
            st.session_state.page = "interview"
            st.session_state["current_question"] = 0
            st.session_state["answers"] = []

    def interview_page(self):
        if "current_question" not in st.session_state or st.session_state.current_question >= len(st.session_state.questions):
            st.session_state.page = "feedback"
            return
        
        question_idx = st.session_state.current_question
        st.title(f"Question {question_idx + 1}")
        
        audio_path = st.session_state.audio_paths[question_idx]
        st.audio(audio_path, format="audio/mp3")
        
        st.write("Record your answer below:")
        audio_answer = st.audio_input("Answer", type="file")
        
        if st.button("Next Question"):
            if audio_answer:
                transcript = self.assistant.audio_to_text(audio_answer)
                st.session_state["answers"].append(transcript)
                st.session_state.current_question += 1

    def feedback_page(self):
        st.title("Interview Feedback")
        feedback_list = [
            self.assistant.generate_feedback(q, a)
            for q, a in zip(st.session_state["questions"], st.session_state["answers"])
        ]
        feedback_text = "\n\n".join(feedback_list)
        
        st.write(feedback_text)
        
        # Send feedback via email
        if send_feedback_email(self.user_info["email"], self.user_info["name"], feedback_text):
            st.success("Feedback sent to your email!")

# Run the application
if __name__ == "__main__":
    app = InterviewApp()
    app.run()
