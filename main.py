"""CLI entrypoint for interacting with the AutoStream LangGraph agent."""

from __future__ import annotations

from agent_template import build_graph
from state import ConversationState

app, config = build_graph()


def run_agent() -> None:
    """Simple REPL loop for the demo video walkthrough."""

    print("--- AutoStream AI Agent is Online ---")
    current_state: ConversationState = {
        "messages": [],
        "user_info": {"name": "", "email": "", "platform": "", "target_plan": ""},
        "intent": "",
    }

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            print("Agent: I didn't catch that. Could you rephrase?")
            continue

        current_state["messages"].append({"role": "user", "content": user_input})

        output = app.invoke(current_state, config=config)
        current_state = output  # graph returns the updated conversation state

        reply = current_state["messages"][-1]["content"] if current_state["messages"] else ""
        print(f"Agent: {reply}")


if __name__ == "__main__":
    run_agent()
