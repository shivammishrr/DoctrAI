import streamlit as st
from dotenv import load_dotenv
import html
import uuid
import time
from queue import Queue

# Load environment variables from .env file
load_dotenv()

# Import core components AFTER dotenv load
from core.doctor_ai import DoctorAI

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="DoctrAI - Medical Assistant",
    page_icon="🩺",
    layout="wide"
)

# This is the comprehensive and enhanced CSS for a polished UI.
CUSTOM_CSS = """
<style>
    /* Main app styling */
    .main {
        background-color: #0f172a; /* Dark blue-gray */
        color: #e2e8f0; /* Light gray for text */
    }
    .stApp {
        background-color: #0f172a;
    }
    
    /* Ensure all text is light by default */
    h1, h2, h3, h4, h5, h6, p, span, div, label, li {
        color: #e2e8f0 !important;
    }

    /* Chat bubble styling */
    .stChatMessage {
        background: #1e293b; /* Dark slate */
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    /* Differentiate user messages */
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: #1e40af; /* Darker blue for user */
    }
    .stChatMessage[data-testid="stChatMessage-user"] div[data-testid="stMarkdownContainer"] p,
    .stChatMessage[data-testid="stChatMessage-user"] div[data-testid="stMarkdownContainer"] li {
        color: white !important;
    }
    /* System message styling for toggle info */
     .stChatMessage[data-testid="stChatMessage-assistant"] div[data-testid="stMarkdownContainer"] .system-info {
        font-style: italic;
        font-size: 0.9rem;
        color: #94a3b8 !important;
        background-color: #27344e;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af, #1e3a8a);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        margin: 0; font-size: 2.7rem; font-weight: 700;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    .main-header p {
        margin-top: 0.7rem; font-size: 1.2rem; opacity: 0.95;
        color: white !important;
    }

    /* Card styling */
    .feature-card {
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 12px; padding: 1.8rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6; /* Blue accent */
    }
    .feature-card h3 {
        color: #60a5fa !important; /* Lighter blue for card titles */
        margin-top: 0; font-size: 1.5rem; font-weight: 600;
    }
    
    /* Research Status & Result Containers */
    .research-status-container {
        background: #1e293b;
        border-radius: 10px; padding: 1.5rem; margin-top:1rem; margin-bottom: 1.5rem;
        border-left: 4px solid #f97316; /* Orange for 'in-progress' */
    }
     .research-status-container .status-header {
        font-size: 1.1rem; font-weight: bold; color: #f97316; margin-bottom: 1rem;
    }
    .research-status-container .status-section-title {
        color: #94a3b8; margin-top: 1rem; margin-bottom: 0.5rem; font-weight: bold;
    }
    .research-status-container ul { padding-left: 20px; }
    
    .result-container {
        background: #1e293b;
        border-radius: 12px; padding: 1.8rem; margin-top: 1.5rem;
        border-left: 6px solid #4ade80; /* Green accent for success */
    }
    .result-container h1, .result-container h2, .result-container h3 { color: #60a5fa !important; }

    .guidance-box {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    /* Footer styling */
    .footer {
        text-align: center; margin-top: 3rem; padding-top: 2rem;
        border-top: 1px solid #334155;
    }
    .footer p, .footer strong {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- App State and Initialization ---
@st.cache_resource
def get_doctor_ai_instance():
    """Returns a cached instance of the main DoctorAI class."""
    return DoctorAI()

doctor_ai = get_doctor_ai_instance()


# --- UI Helper & Chat Logic Functions ---

def initialize_chat_session(session_key_prefix):
    """Initializes session state keys for a new chat instance if they don't exist."""
    if f"{session_key_prefix}_session_id" not in st.session_state:
        st.session_state[f"{session_key_prefix}_session_id"] = str(uuid.uuid4())
        st.session_state[f"{session_key_prefix}_messages"] = []
        st.session_state[f"{session_key_prefix}_is_researching"] = False
        st.session_state[f"{session_key_prefix}_research_queue"] = None
        # Add this line to store previous progress state
        st.session_state[f"{session_key_prefix}_last_progress_state"] = {}

def report_progress_to_ui(placeholder, message: str, status: str):
    """Renders the detailed research progress inside a placeholder - FLICKER-FREE VERSION."""
    current_state = doctor_ai.research_orchestrator.current_research_state
    safe_message = html.escape(message)
    
    # Create new state snapshot
    new_state = {
        'message': safe_message,
        'status': html.escape(status),
        'questions': current_state.get("generated_questions", []),
        'current_question': current_state.get("current_question_researching", ""),
        'tool_calls': current_state.get("tool_calls_for_current_question", [])
    }
    
    # Get the key prefix from the current active tab to access stored state
    active_key = None
    for key in ["symptom", "medication", "lifestyle"]:
        if st.session_state.get(f"{key}_is_researching", False):
            active_key = key
            break
    
    # Compare with previous state to avoid unnecessary updates
    if active_key:
        prev_state = st.session_state.get(f"{active_key}_last_progress_state", {})
        
        # Only update if there's a meaningful change
        if (new_state.get('message') != prev_state.get('message') or
            new_state.get('current_question') != prev_state.get('current_question') or
            len(new_state.get('tool_calls', [])) != len(prev_state.get('tool_calls', [])) or
            new_state.get('status') != prev_state.get('status')):
            
            # Store the new state
            st.session_state[f"{active_key}_last_progress_state"] = new_state
            
            # Build the HTML with the new state
            progress_html = f"""
            <div class='research-status-container'>
                <div class='status-header'>🔬 Deep Research in Progress ({new_state['status']})...</div>
                <p><strong>Last Update:</strong> {new_state['message']}</p>
            """
            
            if new_state['questions']:
                progress_html += "<div class='status-section-title'>Generated Research Questions:</div><ul>" + "".join(f"<li>{html.escape(q)}</li>" for q in new_state['questions']) + "</ul>"
            
            if new_state['current_question']:
                progress_html += f"<div class='status-section-title'>Currently Researching:</div><p><em>{html.escape(new_state['current_question'])}</em></p>"
            
            if new_state['tool_calls']:
                progress_html += "<div class='status-section-title'>Tool Activity:</div><ul>" + "".join(f"<li><strong>Tool:</strong> {html.escape(tc.get('name', 'N/A'))} | <strong>Status:</strong> {html.escape(tc.get('status', 'pending'))}</li>" for tc in new_state['tool_calls']) + "</ul>"
            
            progress_html += "</div>"
            
            # Update the placeholder only when there's actual change
            placeholder.markdown(progress_html, unsafe_allow_html=True)
    else:
        # Fallback to original behavior if no active key found
        progress_html = f"""
        <div class='research-status-container'>
            <div class='status-header'>🔬 Deep Research in Progress ({html.escape(status)})...</div>
            <p><strong>Last Update:</strong> {safe_message}</p>
        """
        if qs := current_state.get("generated_questions"):
            progress_html += "<div class='status-section-title'>Generated Research Questions:</div><ul>" + "".join(f"<li>{html.escape(q)}</li>" for q in qs) + "</ul>"
        if current_q := current_state.get("current_question_researching"):
            progress_html += f"<div class='status-section-title'>Currently Researching:</div><p><em>{html.escape(current_q)}</em></p>"
        if tool_calls := current_state.get("tool_calls_for_current_question"):
            progress_html += "<div class='status-section-title'>Tool Activity:</div><ul>" + "".join(f"<li><strong>Tool:</strong> {html.escape(tc.get('name', 'N/A'))} | <strong>Status:</strong> {html.escape(tc.get('status', 'pending'))}</li>" for tc in tool_calls) + "</ul>"
        progress_html += "</div>"
        placeholder.markdown(progress_html, unsafe_allow_html=True)

def render_chat_interface(key_prefix: str, chat_title: str, welcome_message: str):
    """Renders the UI based on the current state. This function contains NO polling logic."""
    initialize_chat_session(key_prefix)
    
    st.markdown(f"<div class='feature-card'><h3>{chat_title}</h3><p>{welcome_message.splitlines()[1]}</p></div>", unsafe_allow_html=True)
    
    session_id = st.session_state[f"{key_prefix}_session_id"]
    messages = st.session_state[f"{key_prefix}_messages"]
    is_researching = st.session_state[f"{key_prefix}_is_researching"]

    if not messages:
        messages.append({"role": "assistant", "content": welcome_message.splitlines()[0]})

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    research_placeholder = st.empty()

    if is_researching:
        research_queue = st.session_state.get(f"{key_prefix}_research_queue")
        
        last_progress_message = None
        is_complete = False
        final_report = None

        if research_queue:
            while not research_queue.empty():
                update = research_queue.get_nowait()
                if update["type"] == "progress":
                    last_progress_message = update
                elif update["type"] == "complete":
                    is_complete = True
                    final_report = update['report']
                    break
        
        if is_complete:
            final_report_html = f"<div class='result-container'>{final_report}</div>"
            messages.append({"role": "assistant", "content": final_report_html})
            
            system_message = "The deep research has concluded. Thank the user and ask if they have any follow-up questions about the detailed report provided."
            response = doctor_ai.process_turn(session_id, system_message, persona=key_prefix)
            if response and response.get("content"):
                messages.append({"role": "assistant", "content": response.get("content")})
            
            st.session_state[f"{key_prefix}_is_researching"] = False
            st.session_state[f"{key_prefix}_research_queue"] = None
            research_placeholder.empty()

        elif last_progress_message:
            with research_placeholder.container():
                report_progress_to_ui(st, last_progress_message["message"], last_progress_message["status"])
    
    input_container = st.container()
    with input_container:
        deep_research_on = st.toggle(
            "Enable Deep Research", 
            key=f"{key_prefix}_deep_research_toggle",
            help="Performs an in-depth, multi-source investigation instead of a quick conversational answer.", 
            disabled=is_researching
        )
        if deep_research_on and not is_researching:
            st.markdown(
                "<div class='guidance-box'>💡 <b>Deep Research Mode is ON.</b> Your next message will be used as a direct query for an in-depth investigation.</div>",
                unsafe_allow_html=True
            )
        prompt = st.chat_input("Your message...", disabled=is_researching, key=f"chat_input_{key_prefix}")

    if prompt:
        messages.append({"role": "user", "content": prompt})
        if deep_research_on:
            st.session_state[f"{key_prefix}_is_researching"] = True
            info_message = f"<div class='system-info'>✅ Deep Research initiated for: \"{html.escape(prompt)}\". The investigation will begin shortly...</div>"
            messages.append({"role": "assistant", "content": info_message})
            progress_queue = Queue()
            st.session_state[f"{key_prefix}_research_queue"] = progress_queue
            doctor_ai.start_deep_research(session_id, prompt, key_prefix, progress_queue)
        else:
            with st.spinner("Thinking..."):
                response = doctor_ai.process_turn(session_id, prompt, key_prefix)
            if response and response.get("content"):
                 messages.append({"role": "assistant", "content": response["content"]})
        st.rerun()

# --- Main App Layout ---
st.markdown("""
<div class="main-header">
    <h1>🩺 DoctrAI Medical Assistant</h1>
    <p>Your AI-powered medical research and advice companion</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Medication Information", "Lifestyle Recommendations"])

with tab1:
    render_chat_interface(key_prefix="symptom", chat_title="🔍 Symptom Checker", welcome_message="""Hello! I'm DoctorAI.
Describe your symptoms, or use the 'Deep Research' toggle for a detailed investigation into a condition.""")
with tab2:
    render_chat_interface(key_prefix="medication", chat_title="💊 Medication Information", welcome_message="""Hello! I'm DoctorAI.
Ask about a medication, or use 'Deep Research' for a comprehensive report on it.""")
with tab3:
    render_chat_interface(key_prefix="lifestyle", chat_title="🌱 Lifestyle Recommendations", welcome_message="""Hello! I'm DoctorAI.
For which condition do you need lifestyle advice? Or, research a topic in depth.""")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This tool provides AI-generated information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
</div>
""", unsafe_allow_html=True)

# --- Global Polling Controller ---
if any(st.session_state.get(f"{key}_is_researching") for key in ["symptom", "medication", "lifestyle"]):
    time.sleep(1)
    st.rerun()