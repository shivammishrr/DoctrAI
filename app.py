import streamlit as st
from dotenv import load_dotenv
import html

# Load environment variables from .env file
load_dotenv()

# Import core components AFTER dotenv load
from core.doctor_ai import DoctorAI

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="DoctrAI - Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS (adapted from original, can be refined)
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
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #e2e8f0 !important; /* Ensure all text is light */
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af, #1e3a8a); /* Blue gradient */
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
        background: linear-gradient(to right, #1e293b, #0f172a); /* Darker gradient */
        border-radius: 12px; padding: 1.8rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6; /* Blue accent */
    }
    .feature-card h3 {
        color: #60a5fa !important; /* Lighter blue for card titles */
        margin-top: 0; font-size: 1.5rem; font-weight: 600;
    }

    /* Input fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 8px; border: 2px solid #334155; /* Slate border */
        padding: 0.7rem; font-size: 1.05rem;
        background-color: #1e293b; /* Dark input background */
        color: #e2e8f0 !important; /* Light text in input */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6; /* Blue border on focus */
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #3b82f6, #60a5fa); /* Blue gradient button */
        color: white !important; border-radius: 8px; border: none;
        padding: 0.7rem 1.5rem; font-weight: 600; font-size: 1.05rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        transform: translateY(-2px);
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.25);
    }

    /* Research Status Area */
    .research-status-container {
        background: #1e293b; /* Dark background for status */
        border-radius: 10px; padding: 1.5rem; margin-top:1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #3b82f6;
    }
    .research-status-container .status-header {
        font-size: 1.4rem; font-weight: 600; color: #60a5fa !important;
        margin-bottom: 1rem; padding-bottom: 0.8rem;
        border-bottom: 1px solid #334155;
    }
    .research-status-container .status-message {
        font-size: 1.05rem; margin-bottom: 0.8rem; color: #cbd5e1 !important;
    }
    .research-status-container .status-section-title {
        font-weight: 600; color: #93c5fd !important; /* Light blue for section titles */
        margin-top: 1rem; margin-bottom: 0.5rem;
    }
    .research-status-container ul {
        list-style-type: none; padding-left: 0;
    }
    .research-status-container ul li {
        background-color: #27344e; /* Slightly lighter than container */
        padding: 0.8rem; border-radius: 6px; margin-bottom: 0.5rem;
        border-left: 3px solid #60a5fa;
        font-size: 0.95rem;
    }
    .research-status-container .tool-call-item strong {
        color: #60a5fa !important;
    }
    .research-status-container .critique-section {
        margin-top:1rem; padding:1rem; background-color: #27344e; border-radius:6px;
    }

    /* Result container */
    .result-container {
        background: linear-gradient(to right, #1e293b, #0f172a);
        border-radius: 12px; padding: 1.8rem; margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border-left: 6px solid #3b82f6;
        white-space: pre-wrap; /* Preserve formatting of LLM output */
        word-wrap: break-word;
    }

    /* Footer */
    .footer {
        background-color: #1e293b; border-radius: 10px; padding: 1rem;
        margin-top: 3rem; text-align: center; font-size: 0.9rem;
        color: #94a3b8 !important; border-top: 1px solid #334155;
    }
    .footer strong, .footer p {
         color: #94a3b8 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; background-color: #0f172a; padding: 10px 10px 0 10px;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b; border-radius: 10px 10px 0 0;
        padding: 12px 20px; height: auto; font-weight: 500; font-size: 1.05rem;
        color: #94a3b8 !important; /* Light gray for inactive tab text */
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to bottom, #3b82f6, #2563eb) !important;
        color: white !important; font-weight: 600 !important;
        box-shadow: 0 -4px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1e293b; border-radius: 0 0 10px 10px;
        padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Global placeholder for research status updates
research_status_placeholder = st.empty()

# --- App State and Initialization ---
def get_doctor_ai():
    if 'doctor_ai' not in st.session_state:
        st.session_state.doctor_ai = DoctorAI()
    return st.session_state.doctor_ai

doctor_ai_instance = get_doctor_ai()

# --- Progress Reporting Function ---
def report_progress_to_ui(message: str, status: str):
    """ Reports progress from the research orchestrator to the Streamlit UI. """
    # This function is called by the orchestrator.
    # We update the placeholder based on the detailed state from DoctorAI.
    current_state = doctor_ai_instance.get_current_research_state()
    
    # Escape HTML in messages to prevent injection if message content is unexpected
    safe_message = html.escape(message)
    
    progress_html = "<div class=\"research-status-container\">"
    progress_html += f"<div class=\"status-header\">Deep Research Progress ({html.escape(status)})</div>"
    progress_html += f"<div class=\"status-message\"><strong>Last Update:</strong> {safe_message}</div>"

    if current_state.get("initial_query"):
        progress_html += f"<p><strong>Initial Query:</strong> {html.escape(current_state['initial_query'])}</p>"

    if current_state.get("generated_questions"):
        progress_html += "<div class=\"status-section-title\">Generated Research Questions:</div><ul>"
        for i, q in enumerate(current_state["generated_questions"]):
            progress_html += f"<li>{i+1}. {html.escape(q)}</li>"
        progress_html += "</ul>"
    
    current_q_researching = current_state.get("current_question_researching")
    if current_q_researching:
        q_idx = current_state.get("current_question_processing_index", 0)
        total_q = current_state.get("total_questions_to_process", len(current_state.get("generated_questions", [])))
        progress_html += f"<div class=\"status-section-title\">Currently Researching Question {q_idx+1}/{total_q}:</div>"
        progress_html += f"<p><em>{html.escape(current_q_researching)}</em></p>"

        tool_calls = current_state.get("tool_calls_for_current_question", [])
        if tool_calls:
            progress_html += "<div class=\"status-section-title\">Tool Activity:</div><ul>"
            for tc in tool_calls:
                tool_name = html.escape(tc.get("name", "N/A"))
                tool_args = html.escape(str(tc.get("args", "")))
                tool_status = html.escape(tc.get("status", "pending"))
                result_preview = html.escape(tc.get("result_preview", ""))
                progress_html += f"<li class=\"tool-call-item\"><strong>Tool:</strong> {tool_name} | <strong>Args:</strong> {tool_args} | <strong>Status:</strong> {tool_status}"
                if result_preview and tool_status == "success":
                    progress_html += f"<br><em>Preview: {result_preview}</em>"
                progress_html += "</li>"
            progress_html += "</ul>"
    
    if current_state.get("critique_complete"):
        progress_html += "<div class=\"status-section-title critique-section\">Critique Phase:</div>"
        critique_content = current_state.get("critique_content", "Critique in progress or not yet available.")
        progress_html += f"<p>{html.escape(critique_content[:500])}{'...' if len(critique_content) > 500 else ''}</p>"

    if current_state.get("final_report_generated"):
        progress_html += "<div class=\"status-section-title\" style=\"color: #4CAF50 !important;\">Final Report Generated!</div>"
    elif status == "error":
         progress_html += f"<div class=\"status-section-title\" style=\"color: #F44336 !important;\">An Error Occurred.</div>"

    progress_html += "</div>"
    research_status_placeholder.markdown(progress_html, unsafe_allow_html=True)

# Set the callback for the DoctorAI instance
doctor_ai_instance.set_progress_callback(report_progress_to_ui)

# --- UI Layout ---
st.markdown("""
<div class="main-header">
    <h1>🩺 DoctrAI Medical Assistant</h1>
    <p>Your AI-powered medical research and advice companion</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Symptom Checker", "Medication Information", "Lifestyle Recommendations"])

with tab1:
    st.markdown("""
    <div class="feature-card">
        <h3>🔍 Symptom Checker</h3>
        <p>Describe your symptoms for general advice or deep research.</p>
    </div>
    """, unsafe_allow_html=True)
    symptoms = st.text_area("Describe your symptoms:", height=150, key="symptoms_input",
                            placeholder="Example: Persistent headache for 3 days, mild fever, fatigue...")
    deep_research_symptoms = st.checkbox("Enable Deep Research", key="symptoms_deep_research",
                                         help="Uses multiple AI agents and tools for a comprehensive analysis. Takes longer.")
    if st.button("Get Medical Advice", key="symptoms_button"):
        if symptoms:
            research_status_placeholder.empty() # Clear previous status
            with st.spinner("Processing your request..."):
                advice = doctor_ai_instance.get_medical_advice(symptoms, deep_research=deep_research_symptoms)
                if not deep_research_symptoms: # Clear status if it was a simple query and not handled by callback
                    research_status_placeholder.empty()
                st.markdown(f"<div class=\"result-container\">{html.escape(advice)}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please describe your symptoms first.")

with tab2:
    st.markdown("""
    <div class="feature-card">
        <h3>💊 Medication Information</h3>
        <p>Get details on medications or conduct in-depth research.</p>
    </div>
    """, unsafe_allow_html=True)
    medication = st.text_input("Enter medication name:", key="medication_input", 
                               placeholder="Example: Ibuprofen, Amoxicillin, Lisinopril...")
    deep_research_med = st.checkbox("Enable Deep Research", key="medication_deep_research",
                                      help="Uses multiple AI agents and tools for comprehensive information. Takes longer.")
    if st.button("Get Medication Info", key="medication_button"):
        if medication:
            research_status_placeholder.empty()
            with st.spinner("Processing your request..."):
                info = doctor_ai_instance.get_medication_info(medication, deep_research=deep_research_med)
                if not deep_research_med:
                    research_status_placeholder.empty()
                st.markdown(f"<div class=\"result-container\">{html.escape(info)}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a medication name.")

with tab3:
    st.markdown("""
    <div class="feature-card">
        <h3>🌱 Lifestyle Recommendations</h3>
        <p>Receive lifestyle advice for health conditions, with optional deep research.</p>
    </div>
    """, unsafe_allow_html=True)
    condition = st.text_input("Enter health condition:", key="condition_input", 
                                placeholder="Example: Type 2 Diabetes, Hypertension, Asthma...")
    deep_research_lifestyle = st.checkbox("Enable Deep Research", key="lifestyle_deep_research",
                                            help="Uses multiple AI agents and tools for comprehensive advice. Takes longer.")
    if st.button("Get Lifestyle Advice", key="lifestyle_button"):
        if condition:
            research_status_placeholder.empty()
            with st.spinner("Processing your request..."):
                advice = doctor_ai_instance.get_lifestyle_advice(condition, deep_research=deep_research_lifestyle)
                if not deep_research_lifestyle:
                    research_status_placeholder.empty()
                st.markdown(f"<div class=\"result-container\">{html.escape(advice)}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a health condition.")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This tool provides AI-generated information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
    <p>&copy; 2024 DoctrAI - Advanced Medical AI. All Rights Reserved.</p>
</div>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    # This allows the app to be run directly with `python app.py`
    # However, the standard way is `streamlit run app.py`
    pass # No specific action needed here for direct run in this setup 