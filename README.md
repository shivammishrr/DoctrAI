# DoctrAI - Medical Assistant

DoctrAI is an AI-powered medical research and advice companion built with Streamlit and Groq.

## Features

-   Symptom Checker: Provides preliminary medical assessment based on symptoms.
-   Medication Information: Offers detailed information about medications.
-   Lifestyle Recommendations: Gives personalized lifestyle advice for managing health conditions.
-   Deep Research Capability: Utilizes multiple research tools (Tavily, ArXiv, Wikipedia, Google Search) orchestrated by an LLM for comprehensive information gathering.

## Project Structure

```
Medical_AI_Assistant/
├── .env                   # For API keys (GROQ_API_KEY, TAVILY_API_KEY, etc.)
├── README.md
├── requirements.txt
├── app.py                 # Main Streamlit application
└── core/
    ├── __init__.py
    ├── doctor_ai.py         # DoctorAI class
    ├── model_manager.py     # ModelManager class
    ├── research_orchestrator.py # MedicalResearchOrchestrator class
    ├── tool_definitions.py  # Schemas for tools (Groq format)
    └── tool_functions.py    # Python implementations of the tools
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Medical_AI_Assistant
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Create a `.env` file in the root directory (`Medical_AI_Assistant/`) and add your API keys:
    ```env
    GROQ_API_KEY="your_groq_api_key"
    TAVILY_API_KEY="your_tavily_api_key"
    # Add any other API keys if needed
    ```

## Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
```

The application should open in your web browser.

## Disclaimer

This tool provides research information only and should not replace professional medical advice. Always consult qualified healthcare providers for diagnosis and treatment.
