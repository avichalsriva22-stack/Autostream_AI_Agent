"""LangGraph application assembly with LLM selection and checkpointed memory."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from nodes import (
    classify_intent_node,
    information_gathering_node,
    lead_capture_node,
    missing_user_profile_field,
    pricing_rag_node,
)
from state import ConversationState

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints" / "agent.sqlite"


def select_llm(preferred: Optional[str] = None) -> BaseChatModel:
    """Return a chat model using OpenAI or Google Generative AI credentials."""

    load_dotenv()
    preference = (preferred or os.getenv("LLM_PROVIDER", "")).strip().lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if preference in {"google", "gemini"} and google_key:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
    if preference in {"openai", "gpt"} and openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    if google_key:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", convert_system_message_to_human=True)
    if openai_key:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    raise RuntimeError(
        "No LLM credentials found. Set OPENAI_API_KEY or GOOGLE_API_KEY to continue."
    )


def build_agent(
    preferred_llm: Optional[str] = None,
    checkpoint_path: Optional[os.PathLike[str] | str] = None,
) -> Callable:
    """Compile the LangGraph agent with Sqlite-backed memory."""

    llm = select_llm(preferred_llm)

    def classification_node(state: ConversationState) -> ConversationState:
        classify_intent_node(state)
        return state

    def greeting_node(state: ConversationState) -> ConversationState:
        user_message = state["messages"][-1].get("content", "") if state["messages"] else ""
        name = state.get("user_info", {}).get("name", "").strip()
        prompt = (
            "Respond with a short, warm greeting as the AutoStream assistant. "
            "Keep it under two sentences. "
            f"User name: {name or 'friend'}. "
            f"Latest user message: {user_message}"
        )
        try:
            reply = llm.invoke(prompt)
            content = getattr(reply, "content", str(reply))
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("LLM greeting failed, using fallback reply: %s", exc)
            content = "Hi there! Thanks for reaching out to AutoStream - how can I help today?"
        state["messages"].append({"role": "assistant", "content": content})
        return state

    def guard_ready_for_capture(state: ConversationState) -> str:
        missing = missing_user_profile_field(state.get("user_info", {}))
        return "needs_info" if missing else "ready"

    graph = StateGraph(ConversationState)
    graph.add_node("classify", classification_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("pricing", pricing_rag_node)
    graph.add_node("gather_lead_data", information_gathering_node)
    graph.add_node("capture_lead", lead_capture_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        lambda state: state.get("intent", "pricing_inquiry"),
        {
            "greeting": "greeting",
            "pricing_inquiry": "pricing",
            "high_intent_lead": "gather_lead_data",
        },
    )

    graph.add_edge("greeting", END)
    graph.add_edge("pricing", END)

    graph.add_conditional_edges(
        "gather_lead_data",
        guard_ready_for_capture,
        {
            "needs_info": END,
            "ready": "capture_lead",
        },
    )
    graph.add_edge("capture_lead", END)

    if isinstance(checkpoint_path, str) and checkpoint_path.strip() == ":memory:":
        connection = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        resolved_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CHECKPOINT_PATH
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)

    return graph.compile(checkpointer=checkpointer)


__all__ = ["select_llm", "build_agent"]
