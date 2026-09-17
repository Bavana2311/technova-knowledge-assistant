#The system uses conversation memory to understand follow-up questions and only provides information supported by retrieved documents. When the documents do not contain sufficient information, it returns a safe fallback response instead of generating unsupported information.
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory


class ConversationSummaryMemory:
    """Running conversation summary backed by an LLM.

    Lightweight replacement for the deprecated
    ``langchain.memory.ConversationSummaryMemory`` (removed in
    langchain-core v1). Keeps the same ``save_context`` /
    ``load_memory_variables`` interface.
    """

    def __init__(self, llm):
        self.llm = llm
        self.summary = ""

    def save_context(self, inputs: dict, outputs: dict):
        user_input = inputs.get("input", "")
        ai_output = outputs.get("output", "")

        prompt = (
            "Progressively summarize the conversation. Given the current "
            "summary and the new exchange, return an updated summary.\n\n"
            f"Current summary:\n{self.summary or 'None'}\n\n"
            f"New exchange:\nUser: {user_input}\nAssistant: {ai_output}\n\n"
            "Updated summary:"
        )

        response = self.llm.invoke(prompt)
        self.summary = getattr(response, "content", str(response)).strip()

    def load_memory_variables(self, _inputs: dict) -> dict:
        return {"summary": self.summary}


# ============================================================
# SESSION MEMORY STORAGE
# ============================================================

SESSION_MESSAGE_HISTORY = {}
SESSION_SUMMARY_MEMORY = {}
SESSION_RECENT_MEMORY = {}


# ============================================================
# MESSAGE HISTORY
# ============================================================

def get_session_message_history(session_id: str):

    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()

    return SESSION_MESSAGE_HISTORY[session_id]


# ============================================================
# RECENT MEMORY
# ============================================================

def get_session_recent_memory(session_id: str, n: int = 4):

    history = SESSION_RECENT_MEMORY.get(session_id, [])

    return history[-n:]


def update_recent_memory(
    session_id: str,
    user_input: str,
    ai_input: str,
    n: int = 4
):

    if session_id not in SESSION_RECENT_MEMORY:
        SESSION_RECENT_MEMORY[session_id] = []

    SESSION_RECENT_MEMORY[session_id].append(
        {
            "user": user_input,
            "bot": ai_input
        }
    )

    # Keep only the latest n conversations
    SESSION_RECENT_MEMORY[session_id] = (
        SESSION_RECENT_MEMORY[session_id][-n:]
    )


def get_formatted_history(
    session_id: str,
    n: int = 4
) -> str:

    history = get_session_recent_memory(
        session_id,
        n
    )

    if not history:
        return "No recent messages found."

    formatted = []

    for turn in history:

        formatted.append(
            f"User: {turn['user']}\n"
            f"Assistant: {turn['bot']}"
        )

    return "\n\n".join(formatted)


def get_recent_memory(
    session_id: str,
    n: int = 4
) -> str:

    return get_formatted_history(
        session_id,
        n
    )


# ============================================================
# SUMMARY MEMORY
# ============================================================

def get_session_summary_memory(session_id: str):

    if session_id not in SESSION_SUMMARY_MEMORY:

        SESSION_SUMMARY_MEMORY[session_id] = (
            ConversationSummaryMemory(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    temperature=0.2
                )
            )
        )

    return SESSION_SUMMARY_MEMORY[session_id]


def get_recent_summary(session_id: str) -> str:

    memory = get_session_summary_memory(
        session_id
    )

    return memory.load_memory_variables(
        {}
    ).get("summary", "")


# ============================================================
# ADD MEMORY TO RAG CHAIN
# ============================================================

def add_memory_to_chain(
    rag_chain,
    session_id: str,
    enabled: bool = True,
    n: int = 4
):

    if not enabled:

        print("Memory is OFF")

        return rag_chain

    print(
        f"Hybrid Memory is ON for session: "
        f"{session_id}"
    )

    summary_memory = (
        get_session_summary_memory(
            session_id
        )
    )

    def chain_with_summary(
        input_data,
        config=None
    ):

        # Run the existing RAG chain
        response = rag_chain.invoke(
            input_data,
            config=config
        )

        # Extract user question
        question = input_data["question"]

        if isinstance(question, list):

            user_input = question[-1].content

        else:

            user_input = question

        # Extract assistant response
        ai_input = getattr(
            response,
            "content",
            str(response)
        )

        # Save to summary memory
        summary_memory.save_context(
            {"input": user_input},
            {"output": ai_input}
        )

        # Save to message history
        chat_history = (
            get_session_message_history(
                session_id
            )
        )

        chat_history.add_user_message(
            user_input
        )

        chat_history.add_ai_message(
            ai_input
        )

        # Save recent conversation
        update_recent_memory(
            session_id,
            user_input,
            ai_input,
            n
        )

        return response

    runnable_with_summary = RunnableLambda(
        chain_with_summary
    )

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda _: (
            get_session_message_history(
                session_id
            )
        ),
        input_messages_key="question"
    )