# ShopUNow AI Assistant - Cloud Deployment Guide

A multi-department AI chatbot with conversation memory, sentiment analysis, and intelligent routing.

## 🚀 Features

- **Multi-Department Routing**: Automatically routes queries to HR, IT, Facilities, Products, Billing, or Shipping
- **Conversation Memory**: Maintains context across multiple exchanges
- **Sentiment Analysis**: Detects negative sentiment and escalates appropriately
- **Session Management**: Each user gets an isolated conversation session
- **Download Chat History**: Users can export their conversation
- **Responsive UI**: Clean, modern Streamlit interface

## 📋 Prerequisites

- Python 3.9+
- OpenAI API key
- GitHub account
- Streamlit Cloud account (free tier available)

## 🛠️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/shopunow-ai-assistant.git
cd shopunow-ai-assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run Locally

```bash
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` in your browser.

## ☁️ Deploy to Streamlit Cloud

### Step 1: Prepare GitHub Repository

1. Create a new repository on GitHub
2. Push your code:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/shopunow-ai-assistant.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select:
   - Repository: `yourusername/shopunow-ai-assistant`
   - Branch: `main`
   - Main file: `streamlit_app.py`

### Step 3: Configure Secrets

1. In Streamlit Cloud dashboard, go to **App Settings → Secrets**
2. Add your secrets:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-key"

# Optional email settings
ESCALATION_EMAIL = "noreply@shopunow.com"
SUPPORT_EMAIL = "support@shopunow.com"
```

3. Click **Save**

### Step 4: Deploy

Click **Deploy!** and wait for the app to build (2-5 minutes).

## 📁 Project Structure

```
shopunow-ai-assistant/
│
├── streamlit_app.py          # Main Streamlit interface
├── shopunow_agent.py          # Agent logic and core functionality
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .streamlit/
│   └── secrets.toml          # Local secrets (DO NOT COMMIT)
└── .gitignore                # Git ignore rules
```

## 🔧 Configuration

### Agent Settings

Edit `shopunow_agent.py` to customize:

```python
class Config:
    LLM_MODEL = "gpt-4o"              # Change model
    LLM_TEMPERATURE = 0                # Adjust creativity
    EMBEDDING_MODEL = "text-embedding-3-small"
```

### Departments

To add/modify departments, edit the `generate_knowledge_base()` function in `shopunow_agent.py`.

## 📊 Usage

### Basic Chat

1. Open the app
2. Type your question in the chat input
3. Press Enter or click Send
4. Get instant AI-powered responses

### Multi-Department Queries

The agent automatically detects and routes multi-department queries:

- **Single dept**: "What laptops do you have?" → Products
- **Multi dept**: "What laptops and shipping time?" → Products + Shipping

### Download Chat History

1. Click **Download Chat** in the sidebar
2. (Optional) Enter your email
3. Click **Download History**
4. Save the TXT file

## 🔒 Security Notes

- Never commit `.env` or `secrets.toml` files
- Use environment variables for all sensitive data
- Streamlit Cloud secrets are encrypted at rest
- Add `.env` and `secrets.toml` to `.gitignore`

## 📝 .gitignore

Create a `.gitignore` file:

```
# Environment
.env
venv/
env/

# Streamlit
.streamlit/secrets.toml

# Python
__pycache__/
*.py[cod]
*.so
.Python

# Database
*.db
chroma_db/
shopunow_kb.json

# IDE
.vscode/
.idea/
*.swp
*.swo
```

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"

- **Local**: Check your `.env` file
- **Cloud**: Verify secrets in Streamlit Cloud dashboard

### "Module not found"

```bash
pip install -r requirements.txt --upgrade
```

### App crashes on startup

Check Streamlit Cloud logs:
1. Go to app dashboard
2. Click **Manage app**
3. View **Logs** tab

### Slow first response

The agent initializes on first use (10-30 seconds). Subsequent responses are faster.

## 🔄 Updates and Maintenance

### Update Code

```bash
git add .
git commit -m "Your update message"
git push origin main
```

Streamlit Cloud auto-deploys on push.

### Update Dependencies

Edit `requirements.txt`, then:

```bash
pip install -r requirements.txt --upgrade
```

Commit and push changes.

## 📚 API Reference

### Agent Methods

```python
from shopunow_agent import get_agent

agent = get_agent()
agent.initialize()

# Ask a question
result = agent.ask("What laptops do you have?")
print(result['response'])

# Start new chat
session_id = agent.new_chat()

# Get history
history = agent.get_chat_history(session_id)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - feel free to use for your projects!

## 💬 Support

- **Issues**: Open a GitHub issue
- **Email**: support@shopunow.com
- **Phone**: 1-800-SHOPUNOW

## 🎯 Roadmap

- [ ] Add user authentication
- [ ] Implement real email notifications
- [ ] Add analytics dashboard
- [ ] Support file uploads
- [ ] Multi-language support
- [ ] Voice input/output

## 📈 Performance Tips

1. **First Load**: Agent initializes once per session (~30 seconds)
2. **Caching**: Knowledge base is cached after first load
3. **Scaling**: Streamlit Cloud auto-scales based on usage
4. **Optimization**: Use GPT-4o-mini for faster responses (lower cost)

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com)
- [OpenAI API Reference](https://platform.openai.com/docs)

---

**Built with ❤️ by ShopUNow Team**
