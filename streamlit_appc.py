import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
# from audio_recorder_streamlit import audio_recorder
from groq import Groq
from st_audiorec import st_audiorec
from streamlit_TTS import text_to_speech, text_to_audio

# Configure API keys and models
genai.configure(api_key='AIzaSyCEzc2NtaIa3eBMh5QNp1wDaeSCH0OrN-g')
os.environ['GOOGLE_API_KEY'] = "AIzaSyCEzc2NtaIa3eBMh5QNp1wDaeSCH0OrN-g"

GROQ_API_KEY = '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'

client = Groq(api_key=GROQ_API_KEY)

model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    with open(audio_file, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model="whisper-large-v3",
            prompt="Specify context or spelling",  # Optional
            response_format="json",  # Optional
            temperature=0.0  # Optional
        )
    return transcription.text

def extract_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def google_pdf_gemini_embedding(text, type):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model=model, task_type=type, google_api_key=os.environ['GOOGLE_API_KEY'])
        return embedding
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return []

def create_vector_db(texts):
    try:
        embeddings = google_pdf_gemini_embedding(texts, "SEMANTIC_SIMILARITY")
        if not embeddings:
            raise ValueError("Embeddings list is empty.")
        v_db = FAISS.from_texts(texts, embeddings)
        return v_db
    except Exception as e:
        st.error(f"Failed to create vector database: {e}")
        return None

def get_similar_context(v_db, v_user, n):
    if v_user:
        docs = v_db.similarity_search(v_user, k=n)
        return docs

def get_response(query):
    try:
        response = chat_model.generate_content(query, stream=True)
        for res in response:
            if hasattr(res, 'text') and res.text:
                yield res.text
    except ValueError as e:
        st.error(f"Response contains no valid text part: {e}")
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# HTML and CSS for styled title
title_html = """
    <style>
    .title {
        font-size: 70px;
        font-weight: 800;
        color: #c13584; /* Gradient color start */
        background: -webkit-linear-gradient(#4c68d7, #ff6464); /* Gradient background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 50px;
        font-weight: 400;
        color: #333337; /* Subtitle color */
    }
    </style>
    <div class="title">Hello,</div>
    <div class="subtitle">How can I help you today?</div>
    """

st.markdown(title_html, unsafe_allow_html=True)

# Initialize session state variables
if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "v_db" not in st.session_state:
    st.session_state.v_db = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar to switch between text and audio interaction
interaction_mode = st.sidebar.radio("Interaction Mode", ["Text", "Audio"])

# Shared file upload functionality
st.sidebar.title("Chatbot")
pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if pdf and st.sidebar.button("Create Vector Database"):
    with st.spinner("Creating vector database..."):
        texts = text_splitter.split_text(extract_from_pdf(pdf))
        if not texts:
            st.error("No texts were extracted from the PDF.")
        else:
            st.session_state.v_db = create_vector_db(texts)
            st.session_state.pdf = pdf
            st.session_state.texts = texts
            if st.session_state.v_db:
                st.success("Vector database created successfully!")
            else:
                st.error("Failed to create vector database.")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat history cleared!")

if st.sidebar.button("Delete Vector Database"):
    st.session_state.v_db = None
    st.session_state.pdf = None
    st.session_state.texts = None
    st.success("Vector database deleted!")

# Function to process audio input and generate response
def process_audio(audio_data):
    with open("recorded_audio.wav", "wb") as f:
        f.write(audio_data)

    # Transcribe audio using Whisper
    transcription = transcribe_audio("recorded_audio.wav")
    st.session_state.messages.append({"role": "user", "content": transcription})

    # Generate response
    response = ""
    for res in get_response(transcription):
        response += res

    st.session_state.messages.append({"role": "AI", "content": response})
    return response


if interaction_mode == "Text":
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    user_input = st.chat_input("Enter your message:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        placeholder = st.chat_message("AI").empty()
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = get_similar_context(st.session_state.v_db, user_input, 5)
            for doc in similar_context:
                similar_text += doc.page_content

        with st.spinner("Thinking..."):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text

            for response in get_response(combined_input):
                stream_res += response
                placeholder.markdown(stream_res)
            st.session_state.messages.append({"role": "AI", "content": stream_res})
            

elif interaction_mode == "Audio":

    st.title("Voice Interaction")

    # Recording audio
    st.subheader("Record Your Message:")
    wav_audio_data = st_audiorec()


    if wav_audio_data is not None:
        with open("recorded_audio.wav", "wb") as f:
            f.write(wav_audio_data)
        
        # Get the URL for the audio data
        audio_url = st.audio(wav_audio_data, format="audio/wav")
        
        # Use custom HTML and JavaScript to autoplay the audio and make it invisible
        st.markdown(f"""
        <audio id="audio" autoplay>
            <source src="{audio_url}" type="audio/wav">
        </audio>
        <script>
            var audio = document.getElementById('audio');
            audio.style.display = 'none';
            audio.play();
        </script>
        """, unsafe_allow_html=True)

        # Transcribe audio using Whisper
        transcription = transcribe_audio("recorded_audio.webm")
        print(transcription)
        user_input = transcription

        st.session_state.messages.append({"role": "user", "content": user_input})
        # st.chat_message("user").write(user_input)
        # placeholder = st.chat_message("AI").empty()
        similar_text = "You are a Multi Task AI Agent"

        if st.session_state.v_db:
            similar_context = get_similar_context(st.session_state.v_db, user_input, 5)
            for doc in similar_context:
                similar_text += doc.page_content

        with st.spinner("Thinking..."):
            stream_res = ""
            conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
            combined_input += similar_text

            for response in get_response(combined_input):
                stream_res += response
                # placeholder.markdown(stream_res)
            st.session_state.messages.append({"role": "AI", "content": stream_res})
            print(stream_res)
            audioRes = text_to_audio(text=stream_res)
            audio2 = st.audio(audioRes, format="audio/wav")
            # text_to_speech(text=stream_res, language='en')
