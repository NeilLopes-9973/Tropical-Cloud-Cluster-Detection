import subprocess
import sys
import os

# Change to dashboard directory
os.chdir('04_dashboards')

# Run streamlit app
subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'itcc_dashboard_main.py', '--server.port', str(os.environ.get('PORT', 8501)), '--server.address', '0.0.0.0'])
