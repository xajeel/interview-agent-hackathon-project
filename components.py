from transformers import pipeline
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gtts import gTTS
import whisper
from pydub import AudioSegment
from groq import Groq

class InterviewAssistant:
    def __init__(self, questions_path, groq_api_key):
        """Initialize the Interview Assistant with necessary models and data."""
        self.load_questions(questions_path)
        self.setup_models(groq_api_key)
        self.setup_faiss_index()

    def load_questions(self, questions_path):
        """Load and preprocess the questions dataset."""
        self.questions = pd.read_csv(questions_path, encoding='unicode_escape')
        self.questions = self.questions[['Question']]
        
    def setup_models(self, groq_api_key):
        """Initialize all required models."""
        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Text generation model
        self.generation_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            num_beams=1
        )
        
        # Whisper model for speech-to-text
        self.whisper_model = whisper.load_model("base")
        
        # Groq client for LLM feedback
        self.groq_client = Groq(api_key=groq_api_key)

    def setup_faiss_index(self):
        """Create FAISS index for question similarity search."""
        # Generate embeddings for all questions
        embeddings = [self.sentence_model.encode(q) for q in self.questions['Question']]
        self.questions['embedding'] = embeddings
        
        # Create and populate FAISS index
        question_embeddings = np.vstack(self.questions['embedding'].values)
        dimension = question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(question_embeddings)

    def get_similar_questions(self, job_title, job_description, top_k=5):
        """Get similar questions based on job title and description."""
        text = f'{job_title}: {job_description}'
        text_embedding = self.sentence_model.encode(text).reshape(1, -1)
        _, indices = self.index.search(text_embedding, top_k)
        similar_questions = self.questions.iloc[indices[0]]
        return similar_questions[['Question']].to_dict(orient='records')

    def generate_follow_up_questions(self, retrieved_questions):
        """Generate follow-up questions based on retrieved questions."""
        follow_up_questions = []
        for item in retrieved_questions:
            prompt = f"Based on this question '{item['Question']}', suggest another follow-up question for an interview."
            generated = self.generation_model(
                prompt,
                max_length=50,
                num_return_sequences=1,
                truncation=True
            )
            follow_up_text = generated[0]['generated_text'].replace(prompt, "").strip()
            follow_up_questions.append(follow_up_text)
        return follow_up_questions

    def text_to_speech(self, question_text, file_name):
        """Convert text to speech and save as audio file."""
        tts = gTTS(text=question_text, lang='en')
        tts.save(file_name)

    def audio_to_text(self, audio_file):
        """Convert audio to text using Whisper."""
        result = self.whisper_model.transcribe(audio_file)
        return result['text']

    def generate_feedback(self, question, user_answer):
        """Generate feedback on user's interview answer using Groq LLM."""
        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced interview coach specializing in preparing candidates "
                        "for real-world job interviews. Your goal is to provide concise, actionable "
                        "feedback that helps the user improve their answers quickly.\n\n"
                        "Please focus on:\n"
                        "1. Strengths\n"
                        "2. Areas for Improvement\n"
                        "3. What a Best Answer Looks Like\n"
                        "4. English Proficiency\n"
                        "5. Interview Readiness\n"
                        "6. Quick Tips"
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {user_answer}\n\n"
                              "Provide feedback on the quality of the answer, noting strengths "
                              "and suggestions for improvement."
                }
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content

# Example usage:
"""
# Initialize the assistant
assistant = InterviewAssistant(
    questions_path='/path/to/Software Questions.csv',
    groq_api_key='your-groq-api-key'
)

# Get similar questions
questions = assistant.get_similar_questions(
    job_title="Software Engineer",
    job_description="Looking for Python developer with ML experience"
)

# Generate follow-up questions
follow_ups = assistant.generate_follow_up_questions(questions)

# Convert question to speech
assistant.text_to_speech("Tell me about your experience with Python", "question.mp3")

# Convert answer audio to text
answer_text = assistant.audio_to_text("answer.mp3")

# Get feedback on the answer
feedback = assistant.generate_feedback(
    "Tell me about your experience with Python",
    answer_text
)
"""
