from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import json
import time
import os
import random
import re
from typing import Dict, List

app = FastAPI()

# Initialize API key
GROQ_API_KEY = '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'

# Keywords to detect diagnostic questions
DIAGNOSTIC_KEYWORDS = [
    r'what.*(?:wrong|problem|condition|diagnosis)',
    r'(?:tell|explain).*(?:problem|condition)',
    r'(?:what|why).*(?:cause|reason)',
    r'could.*(?:be|have)',
    r'is.*(?:it|this)',
]

# Initialize session state
class SessionState:
    def _init_(self):
        self.case_loaded = False
        self.vectors = None
        self.chat_history = []
        self.case_introduction = ""
        self.asked_if_ready = False
        self.ready_to_start = False
        self.diagnosis_revealed = False
        self.correct_diagnosis = ""
        self.selected_case = None
        self.case_name = None
        self.show_diagnosis_input = False

session_state = SessionState()

def normalize_text(text):
    """Normalize text for comparison."""
    return ' '.join(re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split())

def select_random_case(json_folder):
    """Select a random case file from the JSON folder."""
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    if json_files:
        selected_file = random.choice(json_files)
        return os.path.join(json_folder, selected_file), selected_file
    return None, None

def load_case_data(json_path):
    """Load and process the JSON case file."""
    with open(json_path, 'r') as file:
        case_data = json.load(file)
    
    # Convert case data to format suitable for vector store
    documents = [{"page_content": str(case_data)}]
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts([doc["page_content"] for doc in documents], embeddings)
    
    return vectorstore

def is_diagnostic_question(text):
    """Check if the question is attempting to get diagnostic information."""
    text = text.lower()
    return any(re.search(pattern, text) for pattern in DIAGNOSTIC_KEYWORDS)

def get_patient_response():
    """Get a deflection response when users ask about diagnosis."""
    responses = [
        "I just know it hurts - I'm hoping you can help me understand what's wrong.",
        "That sounds too technical for me. Can you ask me about how I feel instead?",
        "I don't really know about medical stuff. That's why I'm here.",
        "Could you ask me about my symptoms instead?",
    ]
    return random.choice(responses)

def get_chatgroq_response(user_input, is_introduction=False, is_diagnosis=False):
    """Generate response using the ChatGroq model."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    if is_introduction:
        prompt = ChatPromptTemplate.from_template("""
            Provide a very brief patient introduction in exactly 2 lines:
            Hi, I'm [First Name].
            State only the primary symptom in simple terms.
            Don't mention duration, medical terms, or any other details.
            
            Context: {context}
        """)
    elif is_diagnosis:
        prompt = ChatPromptTemplate.from_template("""
            Extract only the primary diagnosis from the case.
            Provide just the basic condition name without any qualifiers.
            
            Context: {context}
        """)
    else:
        if is_diagnostic_question(user_input):
            return get_patient_response(), 0

        prompt = ChatPromptTemplate.from_template("""
            Respond as the patient described in the case. Rules:
            1. Always remember the user is trying to diagnose as per the persona by asking questions, so never ever spill out the diagnosis.
            2. Use only simple language 
            3. Describe only how you feel or what you experience when asked about history or previous data with respect to pain or body part, answer with relevant data.
            4. Keep responses brief and natural
            5. If asked about medical terms, help with clues but never spill out the diagnosis.
            6. Reveal results of x-ray, mri or special tests or any other diagnostic tests if asked just make sure to not spill out the diagnosis.
            
            Context: {context}
            Question: {input}
        """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_input})
    end = time.process_time()

    return response['answer'], end - start

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if not session_state.case_loaded:
                await websocket.send_text("Loading a random case...")
                selected_case_path, case_name = select_random_case('./json_data/')
                if selected_case_path:
                    session_state.selected_case = selected_case_path
                    session_state.case_name = case_name
                    session_state.vectors = load_case_data(selected_case_path)
                    session_state.case_loaded = True
                    await websocket.send_text("Case loaded successfully!")
                    session_state.asked_if_ready = False
                else:
                    await websocket.send_text("No cases found in the specified folder.")
                    return

            if session_state.case_loaded and not session_state.asked_if_ready:
                session_state.chat_history.append({
                    "role": "assistant", 
                    "content": "A case has been selected. Ready to begin?"
                })
                session_state.asked_if_ready = True
                await websocket.send_json(session_state.chat_history)

            if session_state.ready_to_start and not session_state.diagnosis_revealed:
                await websocket.send_text("Submit Diagnosis")

            if session_state.show_diagnosis_input and not session_state.diagnosis_revealed:
                await websocket.send_text("Enter your diagnosis:")
                user_diagnosis = await websocket.receive_text()
                if user_diagnosis:
                    if normalize_text(user_diagnosis) == normalize_text(session_state.correct_diagnosis):
                        await websocket.send_text("Correct diagnosis!")
                    else:
                        await websocket.send_text(f"Incorrect. The correct diagnosis was: {session_state.correct_diagnosis}")
                    await websocket.send_text(f"Case: {session_state.case_name}")
                    session_state.diagnosis_revealed = True

            if data:
                await websocket.send_json({"role": "user", "content": data})
                session_state.chat_history.append({
                    "role": "user", 
                    "content": data
                })

                if not session_state.ready_to_start:
                    if any(word in data.lower() for word in ['yes', 'yeah', 'sure', 'okay', 'ok', 'ready']):
                        session_state.ready_to_start = True
                        await websocket.send_text("Preparing case...")
                        introduction, _ = get_chatgroq_response("", is_introduction=True)
                        session_state.case_introduction = introduction
                        session_state.correct_diagnosis, _ = get_chatgroq_response("", is_diagnosis=True)
                    
                        response_text = f"Let's begin!\n\n{session_state.case_introduction}"
                        await websocket.send_json({"role": "assistant", "content": response_text})
                        session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response_text
                        })
                    else:
                        response_text = "Let me know when you're ready to start."
                        await websocket.send_json({"role": "assistant", "content": response_text})
                        session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response_text
                        })
                else:
                    await websocket.send_text("Thinking...")
                    response, response_time = get_chatgroq_response(data)

                    await websocket.send_json({"role": "assistant", "content": response})
                    await websocket.send_text(f"Response time: {response_time:.2f} seconds")

                    session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })

    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
async def get():
    return HTMLResponse(content=open("index.html").read())

if _name_ == "_main_":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)