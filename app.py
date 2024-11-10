import gradio as gr
from gtts import gTTS
import tempfile
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import whisper
from groq import Groq

# Load data and models
questions = pd.read_csv('non_technical_interview_questions.csv', encoding='unicode_escape')[['Question']]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
question_embeddings = np.vstack([model.encode(q) for q in questions['Question']])
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)
generation_model = pipeline("text2text-generation", model="google/flan-t5-large", num_beams=1)
client = Groq(api_key="GROQ_API_KEY")

# Helper functions
def get_questions(job_title, job_description, top_k=5):
    text = f'{job_title}: {job_description}'
    text_embedding = model.encode(text).reshape(1, -1)
    _, indices = index.search(text_embedding, top_k)
    similar_questions = questions.iloc[indices[0]]
    return similar_questions['Question'].tolist()

def text_to_speech(question_text):
    tts = gTTS(text=question_text, lang='en')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def audio_to_text(audio_file):
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_file)
    return result['text']

def generate_feedback_from_llm(question, user_answer):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced interview coach specializing in preparing candidates for real-world job interviews. "
                    "Your goal is to provide concise, actionable feedback that helps the user improve their answers quickly. "
                    "Please focus on the following points: \n"
                    "1. Strengths: Highlight key strengths in the candidate's answer. Focus on one or two positive aspects (e.g., problem-solving, clear communication). \n"
                    "2. Areas for Improvement: Mention one or two quick improvements, like being more concise, adding specific examples, or avoiding jargon. \n"
                    "3. What a Best Answer Looks Like: Provide a brief example of an ideal answer that addresses the same question with clarity and impact. \n"
                    "4. English Proficiency: Check for major grammar or sentence structure issues and provide quick tips for improvement. \n"
                    "5. Interview Readines: In one sentence, assess if the answer is ready for a real interview or if it needs a little more refinement. \n"
                    "6. Quick Tips: Offer practical, quick tips on how to improve the candidate’s overall interview performance. These could include advice on body language, confidence, tone, or other interview techniques.\n"
                    "Keep your feedback clear and to the point, focusing on the most impactful changes the user can make to improve their interview performance."
                    "Your feedback should always be respectful, professional, and constructive, focused on preparing the candidate to perform confidently and concisely in real-world job interviews."
                )
            },
            {"role": "user", "content": f"Question: {question}\nAnswer: {user_answer}\n\nProvide feedback on the quality of the answer, noting strengths and suggestions for improvement."}
        ],
        model="llama3-8b-8192",
    )
    return response.choices[0].message.content

# Gradio app logic
def start_interview(job_title, job_description):
    question_queue = get_questions(job_title, job_description)
    current_question_index = 0
    question_audio = text_to_speech(question_queue[current_question_index])
    return question_queue, current_question_index, question_audio

def next_question(question_queue, current_question_index):
    current_question_index += 1
    if current_question_index < len(question_queue):
        question_audio = text_to_speech(question_queue[current_question_index])
    else:
        question_audio = None
    return current_question_index, question_audio

def transcribe_and_feedback(answer_audio, question_audio):
    question_text = audio_to_text(question_audio)
    user_answer = audio_to_text(answer_audio)
    feedback = generate_feedback_from_llm(question_text, user_answer)
    return user_answer, feedback

# Gradio UI components
with gr.Blocks() as demo:
    gr.Markdown("### Job Interview Practice App")
    
    with gr.Row():
        job_title = gr.Textbox(label="Job Title", placeholder="e.g., Data Scientist")
        job_description = gr.Textbox(label="Job Description", lines=5, placeholder="Describe the role requirements.")
        start_button = gr.Button("Start Interview")
    
    with gr.Row():
        question_audio = gr.Audio(label="Question", type="filepath", interactive=False)
        next_button = gr.Button("Next Question")

    with gr.Row():
        answer_audio = gr.Audio(label="Your Answer", type="filepath")
    
    with gr.Row():
        response_text = gr.Textbox(label="Transcription of Your Answer", interactive=False)
        feedback_text = gr.Textbox(label="Feedback", interactive=False)
    
    # Define workflow logic
    question_queue = gr.State()
    current_question_index = gr.State()

    start_button.click(start_interview, [job_title, job_description], [question_queue, current_question_index, question_audio])
    next_button.click(next_question, [question_queue, current_question_index], [current_question_index, question_audio])
    
    # Answer transcription and feedback generation
    answer_audio.change(transcribe_and_feedback, [answer_audio, question_audio], [response_text, feedback_text])

demo.launch()
