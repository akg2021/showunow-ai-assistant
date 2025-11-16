# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

### 1. Push Code to GitHub

Your code is ready! You have 2 commits ready to push:

```bash
git push origin main
```

**Note**: You'll need to authenticate. Options:
- Use GitHub Personal Access Token (recommended)
- Set up SSH keys
- Use GitHub CLI: `gh auth login`

### 2. Deploy to Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Configure**:
   - **Repository**: `akg2021/showunow-ai-assistant`
   - **Branch**: `main`
   - **Main file**: `streamlit_app.py`
5. **Click "Deploy!"**

### 3. Configure Secrets

After deployment, go to **App Settings → Secrets** and add:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

**Optional** (if you use email features):
```toml
ESCALATION_EMAIL = "noreply@shopunow.com"
ESCALATION_EMAIL_PASSWORD = "your-email-password"
SUPPORT_EMAIL = "support@shopunow.com"
```

### 4. Your App is Live! 🎉

Streamlit Cloud will:
- Auto-deploy on every push to `main`
- Provide a public URL like: `https://your-app-name.streamlit.app`
- Handle scaling automatically

## What's Included

✅ **Fixed delivery timeline queries** - Now retrieves from vector database
✅ **Improved vector search** - Better retrieval with fallback mechanisms
✅ **Cloud-ready configuration** - Works with Streamlit secrets
✅ **Proper .gitignore** - Sensitive files excluded

## Troubleshooting

### Push Authentication Issues

**Option 1: Personal Access Token**
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use token as password when pushing

**Option 2: SSH Keys**
1. Check if SSH key is added: `cat ~/.ssh/id_ed25519.pub`
2. Add to GitHub: Settings → SSH and GPG keys
3. Use SSH URL: `git remote set-url origin git@github.com:akg2021/showunow-ai-assistant.git`

**Option 3: GitHub CLI**
```bash
gh auth login
git push origin main
```

### Streamlit Cloud Issues

- **Build fails**: Check logs in Streamlit Cloud dashboard
- **API key error**: Verify secrets are set correctly
- **Slow startup**: First load takes 30-60 seconds (normal)

## Next Steps

1. Push your code: `git push origin main`
2. Deploy on Streamlit Cloud
3. Test with: "In how many days will my laptop order be delivered?"
4. Share your app URL!

---

**Ready to deploy?** Just push to GitHub and follow the Streamlit Cloud steps above!

