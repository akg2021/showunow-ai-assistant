cd "$(dirname "$0")"
source venv/bin/activate

# Load environment variables from .env file
set -a
source .env
set +a

# Start Streamlit
streamlit run streamlit_app.py