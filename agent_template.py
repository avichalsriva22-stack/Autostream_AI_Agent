"""Minimal LangGraph wiring of the AutoStream nodes."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, TypeAlias

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from graph import select_llm
from nodes import (
    classify_intent_node,
    information_gathering_node,
    lead_capture_node,
    missing_user_profile_field,
    pricing_rag_node,
    recall_user_profile_node,
    update_user_profile_from_message,
)
from state import ConversationState

_DEFAULT_DB_PATH = Path("checkpoints/template.sqlite")
_DEFAULT_THREAD_ID = "demo-thread"
AgentState: TypeAlias = ConversationState


def route_after_classification(state: AgentState):
    intent = state.get("intent", "")
    if intent == "memory_check":
        return "memory"
    if intent == "lead_captured":
        return END
    if "inquiry" in intent:
        return "researcher"
    if "lead" in intent:
        return "collector"
    return END


def build_graph(
    *,
    preferred_llm: Optional[str] = None,
    checkpoint_path: Optional[str | Path] = None,
):
    """Return a compiled LangGraph app plus config for invocation."""

    load_dotenv()
    llm = select_llm(preferred_llm)

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
        except Exception:  # noqa: BLE001
            content = "Hi there! Thanks for reaching out to AutoStream - how can I help today?"
        state["messages"].append({"role": "assistant", "content": content})
        return state

    def classify_intent(state: AgentState) -> AgentState:
        update_user_profile_from_message(state)
        intent = classify_intent_node(state)
        if intent == "greeting":
            greeting_node(state)
        return state

    def research_knowledge_base(state: AgentState) -> AgentState:
        return pricing_rag_node(state)

    def collect_lead_info(state: AgentState) -> AgentState:
        missing = missing_user_profile_field(state.get("user_info", {}))
        if missing:
            return information_gathering_node(state)
        return lead_capture_node(state)

    workflow = StateGraph(AgentState)

    workflow.add_node("classifier", classify_intent)
    workflow.add_node("researcher", research_knowledge_base)
    workflow.add_node("collector", collect_lead_info)
    workflow.add_node("memory", recall_user_profile_node)

    workflow.add_edge(START, "classifier")
    workflow.add_conditional_edges("classifier", route_after_classification)
    workflow.add_edge("researcher", END)
    workflow.add_edge("collector", END)
    workflow.add_edge("memory", END)

    if isinstance(checkpoint_path, str) and checkpoint_path.strip() == ":memory:":
        connection = sqlite3.connect(":memory:", check_same_thread=False)
    else:
        resolved_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_DB_PATH
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(resolved_path, check_same_thread=False)

    app = workflow.compile(checkpointer=SqliteSaver(connection))
    config = {"configurable": {"thread_id": _DEFAULT_THREAD_ID}}
    return app, config


def main() -> None:
    """Demonstrate a single turn through the graph."""

    app, config = build_graph(checkpoint_path=":memory:")

    state: ConversationState = {
        "messages": [{"role": "user", "content": "How much is the Pro plan?"}],
        "user_info": {"name": "", "email": "", "platform": "", "target_plan": ""},
        "intent": "",
    }

    result = app.invoke(state, config=config)
    reply = result["messages"][-1]["content"] if result["messages"] else ""
    print("Assistant reply:\n", reply)


if __name__ == "__main__":
    main()
