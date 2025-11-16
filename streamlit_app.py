"""
ShopUNow AI Assistant - Streamlit Web Interface

"""

import streamlit as st
import os
from datetime import datetime

# Import agent
from shopunow_agent import get_agent

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="ShopUNow AI Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 15%;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 15%;
        border-left: 4px solid #4caf50;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .stat-box {
        background-color: #ffffff;
        padding: 0.75rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Initialize Session State
# ============================================================================

if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.initialized = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'session_id' not in st.session_state:
    st.session_state.session_id = None

if 'message_count' not in st.session_state:
    st.session_state.message_count = 0

if 'escalation_count' not in st.session_state:
    st.session_state.escalation_count = 0

# ============================================================================
# Initialize Agent (Lazy Loading)
# ============================================================================

def initialize_agent():
    """Initialize agent on first use."""
    if not st.session_state.initialized:
        with st.spinner("Initializing AI Assistant..."):
            try:
                st.session_state.agent = get_agent()
                st.session_state.agent.initialize()
                st.session_state.initialized = True
                st.session_state.session_id = st.session_state.agent.new_chat()
                return True
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                return False
    return True

# ============================================================================
# Header
# ============================================================================

st.markdown('<h1 class="main-header">ShopUNow AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent shopping companion - Ask me anything!</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("Chat Controls")
    
    # New Chat Button
    if st.button("New Chat", use_container_width=True):
        if st.session_state.initialized:
            st.session_state.chat_history = []
            st.session_state.session_id = st.session_state.agent.new_chat()
            st.session_state.message_count = 0
            st.session_state.escalation_count = 0
            st.success("New chat started!")
            st.rerun()
        else:
            st.warning("Please send a message to initialize the agent first.")
    
    st.markdown("---")
    
    # Session Statistics
    st.subheader("Session Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{st.session_state.message_count}</div>
            <div class="stat-label">Messages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{st.session_state.escalation_count}</div>
            <div class="stat-label">Escalations</div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.session_id:
        st.markdown(f"""
        <div class="info-card">
            <small><b>Session ID:</b></small><br>
            <small><code>{st.session_state.session_id[:20]}...</code></small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download Chat History
    st.subheader("Download Chat")
    
    user_email = st.text_input(
        "Email (optional):",
        placeholder="your@email.com",
        help="Provide your email to receive a copy"
    )
    
    if st.button("Download History", use_container_width=True):
        if st.session_state.chat_history:
            # Generate chat history content
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            chat_content = "="*70 + "\n"
            chat_content += "ShopUNow AI Assistant - Chat History\n"
            chat_content += "="*70 + "\n\n"
            
            if st.session_state.session_id:
                chat_content += f"Session ID: {st.session_state.session_id}\n"
            
            chat_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if user_email:
                chat_content += f"Email: {user_email}\n"
            
            chat_content += f"Messages: {len(st.session_state.chat_history)}\n"
            chat_content += "\n" + "="*70 + "\n"
            chat_content += "CONVERSATION\n"
            chat_content += "="*70 + "\n\n"
            
            for i, msg in enumerate(st.session_state.chat_history, 1):
                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                role_label = "You" if msg['role'] == 'user' else "Assistant"
                
                chat_content += f"[{i}] {role_icon} {role_label}:\n"
                chat_content += f"{msg['content']}\n"
                chat_content += "-"*70 + "\n\n"
            
            chat_content += "="*70 + "\n"
            chat_content += "End of conversation\n"
            chat_content += "="*70 + "\n"
            
            # Offer download
            st.download_button(
                label="Download as TXT",
                data=chat_content,
                file_name=f"shopunow_chat_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.success("Ready to download!")
        else:
            st.warning("No chat history to download")
    
    st.markdown("---")
    
    # Help Section
    with st.expander("Help & Info"):
        st.markdown("""
        **How to use:**
        1. Type your question in the chat box
        2. Press Enter or click Send
        3. Get instant AI-powered responses
        
        **I can help with:**
        - Product information
        - Shipping & delivery
        - Billing & payments
        - HR & employee queries
        - IT support
        - Facilities & admin
        
        **Tips:**
        - Be specific in your questions
        - You can ask follow-up questions
        - Use "New Chat" to start fresh
        """)
    
    st.markdown("---")
    st.caption("Powered by ShopUNow AI")
    st.caption("© 2025 ShopUNow Inc.")

# ============================================================================
# Main Chat Area
# ============================================================================

# Display chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        # Welcome message
        st.markdown("""
        <div class="info-card">
            <h3>Welcome to ShopUNow AI Assistant!</h3>
            <p>I'm here to help you with:</p>
            <ul>
                <li>Product information and availability</li>
                <li>Shipping and delivery tracking</li>
                <li>Billing and payment questions</li>
                <li>HR and employee services</li>
                <li>IT support and technical help</li>
                <li>Facilities and administrative queries</li>
            </ul>
            <p><b>Ask me anything to get started!</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display messages
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(
                    f'<div class="chat-message user-message"> <b>You:</b><br>{message["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"> <b>Assistant:</b><br>{message["content"]}</div>', 
                    unsafe_allow_html=True
                )

# ============================================================================
# Chat Input
# ============================================================================

user_input = st.chat_input("Type your message here...", key="chat_input")

if user_input:
    # Initialize agent if needed
    if not initialize_agent():
        st.stop()
    
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    st.session_state.message_count += 1
    
    # Get response from agent
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.agent.ask(
                user_input, 
                session_id=st.session_state.session_id
            )
            
            response = result.get('response', 'I apologize, but I encountered an error.')
            
            # Track escalations
            if result.get('escalated', False):
                st.session_state.escalation_count += 1
            
            # Add assistant response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.session_state.message_count += 1
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': "I apologize, but I encountered an error processing your request. Please try again."
            })
    
    st.rerun()

# ============================================================================
# Quick Action Buttons
# ============================================================================

st.markdown("---")
st.markdown("### Quick Actions")

col1, col2, col3, col4 = st.columns(4)

def handle_quick_action(query: str):
    """Handle quick action button click - process query through agent."""
    # Initialize agent if needed
    if not initialize_agent():
        st.stop()
    
    # Add user message to history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': query
    })
    st.session_state.message_count += 1
    
    # Get response from agent
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.agent.ask(
                query, 
                session_id=st.session_state.session_id
            )
            
            response = result.get('response', 'I apologize, but I encountered an error.')
            
            # Track escalations
            if result.get('escalated', False):
                st.session_state.escalation_count += 1
            
            # Add assistant response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.session_state.message_count += 1
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': "I apologize, but I encountered an error processing your request. Please try again."
            })
    
    st.rerun()

with col1:
    if st.button("Browse Products", use_container_width=True):
        handle_quick_action('What products do you have?')

with col2:
    if st.button("Track Order", use_container_width=True):
        handle_quick_action('How can I track my order?')

with col3:
    if st.button("Billing Help", use_container_width=True):
        handle_quick_action('I have a billing question')

with col4:
    if st.button("Get Support", use_container_width=True):
        handle_quick_action('I need help with an issue')

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>ShopUNow AI Assistant | Powered by GPT-4o | Available 24/7</p>
    <p>For urgent matters, call: 1-800-SHOPUNOW | Email: support@shopunow.com</p>
</div>
""", unsafe_allow_html=True)
