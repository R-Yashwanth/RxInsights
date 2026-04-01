# Pharma RAG - Combined App for Streamlit Cloud
# This runs both backend and frontend in one deployment

import subprocess
import sys
import os
import threading
import time
import requests
import streamlit as st

# Add backend to path
sys.path.append('./backend')

def start_backend():
    \"\"\"Start the FastAPI backend in a separate thread\"\"\"
    try:
        # Change to backend directory
        os.chdir('./backend')
        # Start uvicorn server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'main:app', 
            '--host', '0.0.0.0', 
            '--port', '8000',
            '--reload'
        ], check=True)
    except Exception as e:
        st.error(f\"Failed to start backend: {e}\")

# Start backend in background thread
if 'backend_started' not in st.session_state:
    st.session_state.backend_started = True
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    time.sleep(3)  # Wait for backend to start

# Now run the frontend
exec(open('./frontend/app.py').read())
