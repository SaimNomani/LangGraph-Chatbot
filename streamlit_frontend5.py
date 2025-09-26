import streamlit as st
from langgraph_backend3 import (
    chatbot,
    get_thread_by_id,
    retrieve_all_threads,
    load_conversation,
    get_message_content,
    delete_threads,
    update_thread_title,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    try:
        return uuid.uuid4()
    except Exception as e:
        st.error(f"Failed to generate thread ID: {str(e)}")
        return uuid.uuid4()  # fallback

def reset_chat():
    try:
        thread_id = generate_thread_id()
        st.session_state["thread_id"] = thread_id
        add_thread({
            "thread_id": thread_id,
            "messages": [],
            "thread_title": "New Chat"
        })
        st.session_state["message_history"] = []
        update_thread_title(thread_id=thread_id, new_title="New Chat")
    except Exception as e:
        st.error(f"Error resetting chat: {str(e)}")

def add_thread(thread):
    try:
        if thread["thread_id"] not in [t["thread_id"] for t in st.session_state["chat_threads"]]:
            st.session_state["chat_threads"].append(thread)
    except Exception as e:
        st.error(f"Error adding thread: {str(e)}")

def convert_to_frontend_format(messages):
    frontend_messages = []
    try:
        for msg in messages:
            if hasattr(msg, 'content'):
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                frontend_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                frontend_messages.append(msg)
    except Exception as e:
        st.error(f"Error formatting messages: {str(e)}")
    return frontend_messages

def update_sidebar_threads():
    try:
        backend_threads = retrieve_all_threads()
        st.session_state["chat_threads"] = backend_threads
    except Exception as e:
        st.error(f"Error updating sidebar threads: {str(e)}")

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    try:
        st.session_state["chat_threads"] = retrieve_all_threads()
    except Exception as e:
        st.session_state["chat_threads"] = []
        st.error(f"Error retrieving threads: {str(e)}")

# ✅ Ensure at least one chat exists
if not st.session_state["chat_threads"]:
    reset_chat()

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

if st.sidebar.button("Delete All Chats"):
    try:
        success, msg = delete_threads()
        if success:
            st.sidebar.success(msg)
            st.session_state["chat_threads"] = []
            reset_chat()
            st.rerun()
        else:
            st.sidebar.error(msg)
    except Exception as e:
        st.sidebar.error(f"Error deleting all chats: {str(e)}")

# ============================ Conversations List ============================
st.sidebar.header("My Conversations")
for i, thread in enumerate(st.session_state["chat_threads"][::-1]):
    col1, col2 = st.sidebar.columns([4, 1])

    # Left side → open thread
    with col1:
        btn_text = thread.get("thread_title", "Unnamed Chat")
        if st.button(btn_text, key=f"open_thread_{thread['thread_id']}_{i}"):
            try:
                st.session_state["thread_id"] = thread["thread_id"]
                messages = load_conversation(thread["thread_id"])
                st.session_state["message_history"] = convert_to_frontend_format(messages)
                st.rerun()
            except Exception as e:
                st.error(f"Error opening thread: {str(e)}")

    # Right side → 3-dot menu
    with col2:
        with st.popover("⋮"):
            try:
                # Rename option
                new_name = st.text_input(
                    "Rename chat",
                    value=thread.get("thread_title", "Unnamed Chat"),
                    key=f"rename_input_{thread['thread_id']}_{i}"
                )
                if st.button("✅ Save", key=f"save_btn_{thread['thread_id']}_{i}"):
                    success, msg = update_thread_title(thread["thread_id"], new_name)
                    if success:
                        st.session_state["chat_threads"] = retrieve_all_threads()
                        st.success("Renamed!")
                    else:
                        st.error(msg)
                    st.rerun()

                # Delete option
                if st.button("🗑️ Delete", key=f"delete_btn_{thread['thread_id']}_{i}"):
                    delete_threads(thread["thread_id"])
                    st.session_state["chat_threads"] = retrieve_all_threads()
                    st.rerun()
            except Exception as e:
                st.error(f"Error in chat menu: {str(e)}")

# ============================ Main UI ============================
try:
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
except Exception as e:
    st.error(f"Error rendering chat history: {str(e)}")

user_input = st.chat_input("Type here")

if user_input:
    try:
        # Show user's message
        st.session_state["message_history"].append({"role": "user", "content": user_input})

        thread = get_thread_by_id(st.session_state["thread_id"])
        if thread and thread["thread_title"] == "New Chat":
            new_title = user_input[:20] + "..." if len(user_input) > 20 else user_input
            update_thread_title(thread_id=st.session_state["thread_id"], new_title=new_title)

        with st.chat_message("user"):
            st.text(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_turn",
        }

        # Assistant streaming
        with st.chat_message("assistant"):
            status_holder = {"box": None}

            def ai_only_stream():
                try:
                    for message_chunk, metadata in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Using `{tool_name}` …", expanded=True
                                )
                            else:
                                status_holder["box"].update(
                                    label=f"🔧 Using `{tool_name}` …",
                                    state="running",
                                    expanded=True,
                                )
                        if isinstance(message_chunk, AIMessage):
                            yield message_chunk.content
                except Exception as e:
                    yield f"[Error streaming response: {str(e)}]"

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        # Save assistant message
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

        update_sidebar_threads()
        st.rerun()
    except Exception as e:
        st.error(f"Error during chat handling: {str(e)}")
