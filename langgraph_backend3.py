# backend.py

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

# -------------------
# 1. LLM
# -------------------
try:
    llm = ChatGroq(model="openai/gpt-oss-120b")
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {str(e)}")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": f"Calculator failed: {str(e)}"}

tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    try:
        messages = state.get("messages", [])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        return {"messages": [f"Error generating response: {str(e)}"]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(base_dir, "db")
os.makedirs(name=db_dir, exist_ok=True)
final_db_path = os.path.join(db_dir, "chatbot.db")

try:
    conn = sqlite3.connect(database=final_db_path, check_same_thread=False)
    cursor = conn.cursor()
    checkpointer = SqliteSaver(conn=conn)
except Exception as e:
    raise RuntimeError(f"Database initialization failed: {str(e)}")

def add_column_to_checkpoints_table(column_name, column_type):
    """Add a new column to the checkpoints table if it doesn't already exist."""
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if not cursor.fetchone():
            return False, "Table 'checkpoints' not created yet."

        cursor.execute("PRAGMA table_info(checkpoints)")
        columns = [col[1] for col in cursor.fetchall()]

        if column_name not in columns:
            cursor.execute(f"ALTER TABLE checkpoints ADD COLUMN {column_name} {column_type}")
            conn.commit()
            return True, f"Column '{column_name}' added successfully."
        return True, f"Column '{column_name}' already exists."
    except Exception as e:
        return False, f"Error adding column '{column_name}': {str(e)}"

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)
add_column_to_checkpoints_table("thread_title", "TEXT")

# -------------------
# 7. Helper
# -------------------
def load_conversation(thread_id):
    try:
        CONFIG = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config=CONFIG)
        return state.values.get("messages", [])
    except Exception as e:
        return [{"error": f"Failed to load conversation: {str(e)}"}]

def get_message_content(msg):
    try:
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, dict) and 'content' in msg:
            return msg['content']
        elif isinstance(msg, str):
            return msg
        else:
            return str(msg)
    except Exception as e:
        return f"[Invalid message: {str(e)}]"

def get_thread_by_id(thread_id):
    try:
        cursor.execute("SELECT thread_title FROM checkpoints WHERE thread_id= ?", (str(thread_id),))
        row = cursor.fetchone()
        messages = load_conversation(thread_id)

        if row and row[0]:
            thread_title = row[0]
        elif messages:
            first_message_content = get_message_content(messages[0])
            thread_title = first_message_content[:20] + "..." if len(first_message_content) > 20 else first_message_content
        else:
            thread_title = "New Chat"

        return {"thread_id": thread_id, "messages": messages, "thread_title": thread_title}
    except Exception as e:
        return {"thread_id": thread_id, "messages": [], "thread_title": f"Error: {str(e)}"}

def retrieve_all_threads():
    try:
        all_threads = set()
        all_threads_objs = []
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])

        for thread in list(all_threads):
            messages = load_conversation(thread)
            cursor.execute("SELECT thread_title FROM checkpoints WHERE thread_id= ?", (thread,))
            row = cursor.fetchone()
            if row and row[0]:
                thread_title = row[0]
            else:
                thread_title = "New Chat"
                if messages:
                    first_message_content = get_message_content(messages[0])
                    thread_title = first_message_content[:20] + "..." if len(first_message_content) > 20 else first_message_content

            all_threads_objs.append({"thread_id": thread, "messages": messages, "thread_title": thread_title})

        return all_threads_objs
    except Exception as e:
        return [{"thread_id": None, "messages": [], "thread_title": f"Error: {str(e)}"}]

def delete_threads(thread_id=None):
    """Delete all conversation threads or particular thread from the database"""
    try:
        if thread_id:
            checkpointer.delete_thread(thread_id)
            return True, f"Successfully deleted conversation with ID: {thread_id}"

        all_threads = {cp.config["configurable"]["thread_id"] for cp in checkpointer.list(None)}
        for t_id in all_threads:
            checkpointer.delete_thread(t_id)

        return True, f"Successfully deleted {len(all_threads)} conversations"
    except Exception as e:
        return False, f"Error deleting threads: {str(e)}"

def update_thread_title(thread_id, new_title):
    """Update the thread_title for specific thread_id """
    try:
        cursor.execute("UPDATE checkpoints SET thread_title= ? WHERE thread_id= ?", (new_title, thread_id))
        conn.commit()
        if cursor.rowcount == 0:
            return False, f"No thread found with ID: {thread_id}"
        return True, f"Thread title updated successfully for ID: {thread_id}"
    except Exception as e:
        return False, f"Error updating thread title for ID '{thread_id}': {str(e)}"
