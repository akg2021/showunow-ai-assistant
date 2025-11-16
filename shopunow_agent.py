"""
ShopUNow AI Agent - Cloud-Ready Version
Optimized for Streamlit Cloud Deployment
"""

import os
import json
import uuid
import hashlib
from typing import TypedDict, Literal, Annotated, Sequence, List, Dict, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import chromadb

# Suppress ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration."""
    
    # API Keys (loaded from environment or Streamlit secrets)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Email settings (optional)
    ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL", "noreply@shopunow.com")
    ESCALATION_EMAIL_PASSWORD = os.getenv("ESCALATION_EMAIL_PASSWORD", "")
    SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@shopunow.com")
    
    # Paths
    KB_FILE = "shopunow_kb.json"
    CHROMA_DIR = "./chroma_db"
    
    # Model settings
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0
    EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================================
# State Models
# ============================================================================

class SubQuery(BaseModel):
    """Sub-question for multi-department queries."""
    question: str = Field(description="The sub-question")
    department: Literal[
        'HR Department',
        'IT Support',
        'Facilities & Admin',
        'Products',
        'Shipping & Delivery',
        'Billing & Payments'
    ]
    priority: int = Field(ge=1, description="Priority (1=highest)")


class QueryDecomposition(BaseModel):
    """Query decomposition result."""
    is_multi_department: bool
    sub_queries: List[SubQuery]
    explanation: str


class DepartmentCategory(BaseModel):
    """Department categorization."""
    department: Literal[
        'HR Department',
        'IT Support',
        'Facilities & Admin',
        'Products',
        'Shipping & Delivery',
        'Billing & Payments',
        'Unknown'
    ]
    user_type: Literal['customer', 'employee', 'unknown']


class QuerySentiment(BaseModel):
    """Sentiment analysis."""
    sentiment: Literal['Positive', 'Neutral', 'Negative']


class AgentState(TypedDict):
    """Complete agent state with memory."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    is_multi_department: bool
    sub_queries: List[Dict]
    department_responses: Dict
    user_type: str
    department: str
    sentiment: str
    final_response: str
    escalation_reason: str
    requires_escalation: bool


# ============================================================================
# Knowledge Base Manager
# ============================================================================

class KnowledgeBaseManager:
    """Manages knowledge base generation and loading."""
    
    def __init__(self, llm):
        self.llm = llm
        self.kb_file = Config.KB_FILE
    
    def generate_knowledge_base(self):
        """Generate synthetic FAQ data for all departments."""
        
        departments = [
            {
                'name': 'HR Department',
                'description': 'Employee benefits, leave policies, payroll, performance reviews',
                'user_type': 'Internal Employee'
            },
            {
                'name': 'IT Support',
                'description': 'System issues, software, hardware requests, password resets',
                'user_type': 'Internal Employee'
            },
            {
                'name': 'Facilities & Admin',
                'description': 'Workspace logistics, maintenance, access cards, office supplies',
                'user_type': 'Internal Employee'
            },
            {
                'name': 'Products',
                'description': 'Product details, specifications, availability, pricing',
                'user_type': 'External Customer'
            },
            {
                'name': 'Billing & Payments',
                'description': 'Invoice issues, refunds, payment methods, overcharges',
                'user_type': 'External Customer'
            },
            {
                'name': 'Shipping & Delivery',
                'description': 'Order tracking, returns pickup, damaged goods, shipping delays',
                'user_type': 'External Customer'
            }
        ]
        
        all_knowledge_base = []
        
        for dept in departments:
            prompt = f"""Generate 25 realistic question-answer pairs for {dept['name']} at ShopUNow retail company.
            Description: {dept['description']}
            User Type: {dept['user_type']}
            
            Return ONLY valid JSON array:
            [{{"question": "Q1?", "answer": "A1"}}, {{"question": "Q2?", "answer": "A2"}}]
            """
            
            try:
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Clean JSON
                if content.startswith('```json'):
                    content = content[7:]
                if content.startswith('```'):
                    content = content[3:]
                if content.endswith('```'):
                    content = content[:-3]
                
                qa_pairs = json.loads(content.strip())
                
                for qa in qa_pairs:
                    all_knowledge_base.append({
                        'text': f"Question: {qa['question']} Answer: {qa['answer']}",
                        'metadata': {
                            'department': dept['name'].lower().replace(' ', '_').replace('&', '_'),
                            'user_type': dept['user_type'].lower()
                        }
                    })
            except Exception as e:
                print(f"Error generating KB for {dept['name']}: {e}")
        
        # Save
        with open(self.kb_file, 'w') as f:
            json.dump(all_knowledge_base, f, indent=2)
        
        return all_knowledge_base
    
    def load_or_generate(self):
        """Load existing KB or generate new one."""
        try:
            with open(self.kb_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.generate_knowledge_base()


# ============================================================================
# Vector Database Manager
# ============================================================================

class VectorDBManager:
    """Manages vector database creation and retrieval."""
    
    def __init__(self):
        self.db = None
        self.retriever = None
    
    def create_database(self, knowledge_base):
        """Create vector database with telemetry disabled."""
        
        # Process documents
        processed_docs = [
            Document(
                page_content=doc['text'],
                metadata=doc['metadata']
            )
            for doc in knowledge_base
        ]
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # Configure ChromaDB
        client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        # Create database
        self.db = Chroma.from_documents(
            documents=processed_docs,
            collection_name='shopunow_kb',
            embedding=embeddings,
            persist_directory=Config.CHROMA_DIR,
            client_settings=client_settings
        )
        
        # Create retriever
        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.2}
        )
        
        return self.db, self.retriever


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """Manages user chat sessions."""
    
    def __init__(self):
        self.active_sessions = {}
        self.current_session_id = None
    
    def start_new_chat(self) -> str:
        """Start a new anonymous chat session."""
        session_id = f"chat_{uuid.uuid4().hex[:12]}"
        self.current_session_id = session_id
        
        self.active_sessions[session_id] = {
            'session_id': session_id,
            'started_at': datetime.now(),
            'last_activity': datetime.now(),
            'messages': [],
            'user_email': None,
            'metadata': {}
        }
        
        return session_id
    
    def get_current_session(self) -> str:
        """Get or create current session."""
        if not self.current_session_id or self.current_session_id not in self.active_sessions:
            return self.start_new_chat()
        return self.current_session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to session history."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'role': role,
                'content': content
            })
            self.active_sessions[session_id]['last_activity'] = datetime.now()
    
    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]['messages']
        return []


# ============================================================================
# Agent Nodes
# ============================================================================

class AgentNodes:
    """Agent workflow nodes."""
    
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def check_sentiment(self, state: AgentState) -> AgentState:
        """Check query sentiment for escalation."""
        query = state["user_query"]
        
        prompt = f"""Analyze sentiment of this customer query: "{query}"
        
        Return: Positive, Neutral, or Negative"""
        
        result = self.llm.with_structured_output(QuerySentiment).invoke(prompt)
        sentiment = result.sentiment
        
        if sentiment == "Negative":
            return {
                "sentiment": sentiment,
                "requires_escalation": True,
                "escalation_reason": "Negative sentiment detected"
            }
        
        return {"sentiment": sentiment}
    
    def decompose_query(self, state: AgentState) -> AgentState:
        """Decompose query into department-based sub-queries."""
        
        query = state["user_query"]
        messages = state.get("messages", [])
        
        # Build conversation context
        context = ""
        if messages:
            context = "\n\nRecent conversation:\n"
            for msg in messages[-4:]:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                context += f"{role}: {msg.content[:100]}...\n"
        
        prompt = f"""Analyze this query and decompose it for department routing.

{context}

Current query: {query}

Departments:
- HR Department: Employee benefits, leave, payroll
- IT Support: Tech issues, passwords, software
- Facilities & Admin: Workspace, maintenance
- Products: Product info, pricing, availability
- Billing & Payments: Invoices, refunds
- Shipping & Delivery: Tracking, returns, delivery

IMPORTANT:
- If query has ONE main topic for ONE department: Create ONE sub-query for that department
- If query has MULTIPLE topics for DIFFERENT departments: Create MULTIPLE sub-queries
- ALWAYS create at least one sub-query
- Assign priority: 1 (first/most important), 2, 3, etc.

Return: is_multi_department (true if >1 sub-query), sub_queries list, explanation
"""
        
        result = self.llm.with_structured_output(QueryDecomposition).invoke(prompt)
        
        # Force at least one sub-query if none provided
        if not result.sub_queries:
            result.sub_queries = [
                SubQuery(question=query, department='Products', priority=1)
            ]
        
        is_multi = len(result.sub_queries) > 1
        
        sub_queries_dict = [
            {
                "question": sq.question,
                "department": sq.department,
                "priority": sq.priority
            }
            for sq in result.sub_queries
        ]
        
        return {
            "messages": [HumanMessage(content=query)],
            "is_multi_department": is_multi,
            "sub_queries": sub_queries_dict,
            "department_responses": {},
            "requires_escalation": False
        }
    
    def route_to_departments(self, state: AgentState) -> AgentState:
        """Route queries to departments and collect responses."""
        
        sub_queries = state["sub_queries"]
        department_responses = {}
        failed_count = 0
        
        dept_mapping = {
            'HR Department': 'hr_department',
            'IT Support': 'it_support',
            'Facilities & Admin': 'facilities___admin',
            'Products': 'products',
            'Shipping & Delivery': 'shipping___delivery',
            'Billing & Payments': 'billing___payments'
        }
        
        sorted_queries = sorted(sub_queries, key=lambda x: x['priority'])
        
        for sq in sorted_queries:
            question = sq['question']
            department = sq['department']
            priority = sq['priority']
            
            dept_filter = dept_mapping.get(department)
            
            if not dept_filter:
                failed_count += 1
                department_responses[department] = {
                    "question": question,
                    "response": "Department not available",
                    "success": False,
                    "priority": priority
                }
                continue
            
            try:
                # Retrieve from knowledge base
                self.retriever.search_kwargs["filter"] = {'department': dept_filter}
                docs = self.retriever.invoke(question)
                content = "\n\n".join(doc.page_content for doc in docs)
                
                if not content or len(docs) == 0:
                    failed_count += 1
                    department_responses[department] = {
                        "question": question,
                        "response": "No information available in knowledge base",
                        "success": False,
                        "priority": priority
                    }
                    continue
                
                # Generate response
                prompt = ChatPromptTemplate.from_template(
                    """You are answering for ShopUNow's {department}.

Question: {question}

Knowledge Base Information:
{content}

Provide a clear, focused answer to this specific question.
Be concise and informative.
"""
                )
                
                chain = prompt | self.llm
                response = chain.invoke({
                    "department": department,
                    "question": question,
                    "content": content
                }).content
                
                department_responses[department] = {
                    "question": question,
                    "response": response,
                    "success": True,
                    "priority": priority
                }
            
            except Exception as e:
                failed_count += 1
                department_responses[department] = {
                    "question": question,
                    "response": f"Error: {str(e)}",
                    "success": False,
                    "priority": priority
                }
        
        # Check if all failed - escalate
        if failed_count == len(sub_queries):
            return {
                "department_responses": department_responses,
                "requires_escalation": True,
                "escalation_reason": "No information available in knowledge base"
            }
        
        return {
            "department_responses": department_responses,
            "requires_escalation": False
        }
    
    def compile_response(self, state: AgentState) -> AgentState:
        """Compile responses from departments."""
        
        query = state["user_query"]
        dept_responses = state["department_responses"]
        is_multi = state.get("is_multi_department", False)
        
        # Sort by priority
        sorted_responses = sorted(
            dept_responses.items(),
            key=lambda x: x[1].get('priority', 999)
        )
        
        # Build context for LLM
        context = f"Original query: {query}\n\n"
        context += f"Department responses:\n\n"
        
        for dept, data in sorted_responses:
            status = "✓" if data.get('success') else "✗"
            context += f"[{status}] {dept} (Priority {data.get('priority')})\n"
            context += f"Question: {data['question']}\n"
            context += f"Answer: {data['response']}\n\n"
        
        # Different compilation strategy based on single vs multi
        if is_multi:
            prompt = f"""You are compiling responses from multiple ShopUNow departments.

{context}

Task: Create ONE cohesive, well-structured response that addresses ALL parts of the original query.

Guidelines:
1. Organize information logically by priority
2. Use clear sections or formatting if helpful
3. Make it flow naturally - don't just list separate answers
4. If any department couldn't answer, acknowledge gracefully
5. The user should feel they got ONE complete answer, not separate pieces

Provide a unified, comprehensive response.
"""
        else:
            prompt = f"""You are presenting a response from ShopUNow.

{context}

Task: Present this answer in a clear, helpful, professional manner.

Guidelines:
1. Keep the core information from the department response
2. Make it conversational and friendly
3. If there was no information, acknowledge politely
4. You can slightly rephrase for better flow, but keep the facts accurate

Provide the final response.
"""
        
        compiled = self.llm.invoke(prompt).content
        
        return {
            "messages": [AIMessage(content=compiled)],
            "final_response": compiled
        }
    
    def escalate_to_human(self, state: AgentState) -> AgentState:
        """Escalate to human agent."""
        
        query = state["user_query"]
        reason = state.get("escalation_reason", "Negative sentiment")
        
        ref_number = abs(hash(query)) % 10000
        
        if "negative" in reason.lower() or "sentiment" in reason.lower():
            message = f"""We sincerely apologize. Your query has been escalated.

A specialist will contact you shortly.

Reference: ESC-{ref_number:04d}

Thank you for your patience."""
        else:
            message = f"""Thank you for your query. We're connecting you with a specialist.

Reference: ESC-{ref_number:04d}
Response time: Within 24 hours

For urgent matters: 1-800-SHOPUNOW"""
        
        return {
            "messages": [AIMessage(content=message)],
            "final_response": message
        }


# ============================================================================
# Agent Builder
# ============================================================================

class ShopUNowAgent:
    """Main agent class."""
    
    def __init__(self):
        self.config = Config()
        self.llm = None
        self.agent = None
        self.session_manager = SessionManager()
        self._initialized = False
    
    def initialize(self):
        """Initialize agent (called once)."""
        if self._initialized:
            return
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Load knowledge base
        kb_manager = KnowledgeBaseManager(self.llm)
        knowledge_base = kb_manager.load_or_generate()
        
        # Create vector database
        db_manager = VectorDBManager()
        _, retriever = db_manager.create_database(knowledge_base)
        
        # Build agent graph
        nodes = AgentNodes(self.llm, retriever)
        graph = self._build_graph(nodes)
        
        # Compile with memory
        memory = MemorySaver()
        self.agent = graph.compile(checkpointer=memory)
        
        self._initialized = True
    
    def _build_graph(self, nodes):
        """Build agent workflow graph."""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("check_sentiment", nodes.check_sentiment)
        graph.add_node("decompose_query", nodes.decompose_query)
        graph.add_node("route_to_departments", nodes.route_to_departments)
        graph.add_node("compile_response", nodes.compile_response)
        graph.add_node("escalate_to_human", nodes.escalate_to_human)
        
        # Entry point
        graph.set_entry_point("check_sentiment")
        
        # Routing logic
        def check_sentiment_route(state):
            return "escalate" if state.get("requires_escalation") else "decompose"
        
        def check_escalation_route(state):
            return "escalate" if state.get("requires_escalation") else "compile"
        
        graph.add_conditional_edges(
            "check_sentiment",
            check_sentiment_route,
            {"decompose": "decompose_query", "escalate": "escalate_to_human"}
        )
        
        graph.add_edge("decompose_query", "route_to_departments")
        
        graph.add_conditional_edges(
            "route_to_departments",
            check_escalation_route,
            {"compile": "compile_response", "escalate": "escalate_to_human"}
        )
        
        graph.add_edge("compile_response", END)
        graph.add_edge("escalate_to_human", END)
        
        return graph
    
    def ask(self, query: str, session_id: Optional[str] = None) -> Dict:
        """Ask a question and get response."""
        
        # Ensure initialized
        if not self._initialized:
            self.initialize()
        
        # Get or create session
        if not session_id:
            session_id = self.session_manager.get_current_session()
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Add user message to history
        self.session_manager.add_message(session_id, 'user', query)
        
        # Get conversation history
        try:
            current_state = self.agent.get_state(config)
            messages = current_state.values.get("messages", []) if current_state.values else []
        except:
            messages = []
        
        # Initial state
        initial_state = {
            "user_query": query,
            "messages": messages,
            "is_multi_department": False,
            "sub_queries": [],
            "department_responses": {},
            "sentiment": "Neutral",
            "department": "",
            "user_type": "",
            "escalation_reason": "",
            "requires_escalation": False
        }
        
        # Run agent
        final_event = None
        for event in self.agent.stream(initial_state, config, stream_mode="values"):
            final_event = event
        
        # Get response
        if final_event:
            response = final_event.get('final_response', 'I apologize, but I encountered an error.')
            
            # Add assistant response to history
            self.session_manager.add_message(session_id, 'assistant', response)
            
            return {
                'response': response,
                'session_id': session_id,
                'is_multi_department': final_event.get('is_multi_department', False),
                'sentiment': final_event.get('sentiment', 'Neutral'),
                'escalated': final_event.get('requires_escalation', False)
            }
        
        return {
            'response': 'I apologize, but I encountered an error processing your request.',
            'session_id': session_id,
            'error': True
        }
    
    def new_chat(self) -> str:
        """Start a new chat session."""
        return self.session_manager.start_new_chat()
    
    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session."""
        return self.session_manager.get_chat_history(session_id)


# ============================================================================
# Module-level instance (singleton pattern)
# ============================================================================

_agent_instance = None

def get_agent() -> ShopUNowAgent:
    """Get or create agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ShopUNowAgent()
    return _agent_instance
