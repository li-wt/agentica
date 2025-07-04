import subprocess
subprocess.run(["streamlit", "run", "web_chat_app.py", "--server.port", "8504", "--server.address", "0.0.0.0"])