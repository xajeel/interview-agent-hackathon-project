import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from pathlib import Path
from components import InterviewAssistant

# Configuration
EMAIL_SENDER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-app-specific-password"
GITHUB_LINK = "https://github.com/yourusername/project"
TEAM_WEBSITE = "https://yourteam.com"

# Create necessary directories
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
            questions_path='Software Questions.csv',
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        self.current_question = 0
        self.questions = []
        self.answers = []
        self.user_info = {}

    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(css=self.get_custom_css()) as app:
            # Navigation bar
            with gr.Row(elem_id="navbar"):
                gr.HTML("""
                    <div class="nav-container">
                        <a href="/" class="nav-item">Home</a>
                        <a href="#" class="nav-item" onclick="window.open('""" + GITHUB_LINK + """')">GitHub</a>
                        <a href="#" class="nav-item" onclick="window.open('""" + TEAM_WEBSITE + """')">About Team</a>
                    </div>
                """)
            
            # Welcome Page
            with gr.Tab("Welcome") as welcome_page:
                gr.Markdown("# AI Interview Assistant")
                gr.Markdown("""
                Welcome to the AI Interview Assistant! This application helps you practice 
                for job interviews with AI-generated questions and personalized feedback.
                Simply enter your details, and we'll create a customized interview experience for you.
                """)
                
                # Job trends plot
                gr.Plot(value=generate_mock_job_trends())
                start_btn = gr.Button("Start Interview", variant="primary")
            
            # User Info Page
            with gr.Tab("User Info", visible=False) as info_page:
                with gr.Form():
                    name = gr.Textbox(label="Full Name")
                    email = gr.Textbox(label="Email")
                    job_title = gr.Textbox(label="Desired Job Title")
                    job_desc = gr.Textbox(label="Job Description", lines=3)
                    submit_info = gr.Button("Start")
            
            # Interview Page
            with gr.Tab("Interview", visible=False) as interview_page:
                with gr.Column():
                    status = gr.Markdown("Preparing your interview questions...")
                    audio_player = gr.Audio(label="Question", visible=False)
                    audio_recorder = gr.Audio(source="microphone", label="Your Answer", visible=False)
                    next_btn = gr.Button("Next Question", visible=False)
            
            # Feedback Page
            with gr.Tab("Feedback", visible=False) as feedback_page:
                feedback_text = gr.Markdown()

            def start_interview():
                return {
                    welcome_page: gr.update(visible=False),
                    info_page: gr.update(visible=True)
                }

            def process_user_info(name, email, job_title, job_desc):
                self.user_info = {
                    "name": name,
                    "email": email,
                    "job_title": job_title,
                    "job_desc": job_desc
                }
                
                # Get questions and convert to audio
                questions = self.assistant.get_similar_questions(job_title, job_desc)
                self.questions = [q["Question"] for q in questions]
                
                # Convert questions to audio
                for i, q in enumerate(self.questions):
                    audio_path = f"temp_audio/question_{i}.mp3"
                    self.assistant.text_to_speech(q, audio_path)
                
                return {
                    info_page: gr.update(visible=False),
                    interview_page: gr.update(visible=True),
                    audio_player: gr.update(visible=True, value=f"temp_audio/question_0.mp3"),
                    audio_recorder: gr.update(visible=True),
                    next_btn: gr.update(visible=True),
                    status: gr.update(value=f"Question 1 of {len(self.questions)}")
                }

            def process_answer(audio):
                if audio is None:
                    return
                
                self.answers.append(self.assistant.audio_to_text(audio))
                self.current_question += 1
                
                if self.current_question >= len(self.questions):
                    # Generate feedback
                    all_feedback = []
                    for q, a in zip(self.questions, self.answers):
                        feedback = self.assistant.generate_feedback(q, a)
                        all_feedback.append(feedback)
                    
                    combined_feedback = "\n\n".join(all_feedback)
                    
                    # Send email
                    send_feedback_email(self.user_info["email"], self.user_info["name"], combined_feedback)
                    
                    return {
                        interview_page: gr.update(visible=False),
                        feedback_page: gr.update(visible=True),
                        feedback_text: gr.update(value=combined_feedback)
                    }
                else:
                    return {
                        audio_player: gr.update(value=f"temp_audio/question_{self.current_question}.mp3"),
                        status: gr.update(value=f"Question {self.current_question + 1} of {len(self.questions)}")
                    }

            # Event handlers
            start_btn.click(start_interview)
            submit_info.click(process_user_info, [name, email, job_title, job_desc])
            next_btn.click(process_answer, [audio_recorder])

        return app

    def get_custom_css(self):
        return """
        .nav-container {
            display: flex;
            justify-content: space-around;
            padding: 1rem;
            background-color: #f0f0f0;
            margin-bottom: 2rem;
        }
        .nav-item {
            text-decoration: none;
            color: #2c3e50;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .nav-item:hover {
            background-color: #e0e0e0;
        }
        """

# Launch the application
if __name__ == "__main__":
    app = InterviewApp()
    interface = app.create_interface()
    interface.launch(share=True)
