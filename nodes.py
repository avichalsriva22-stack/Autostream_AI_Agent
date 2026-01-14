
"""Graph nodes for conversation flow control and retrieval."""

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from state import ConversationState


IntentRoute = Literal["greeting", "pricing_inquiry", "high_intent_lead", "memory_check"]

# Keyword groups tuned for simple rule-based intent classification.
_GREETING_PATTERNS = re.compile(r"\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening))\b", re.I)
_PRICING_PATTERNS = re.compile(r"\b(price|pricing|cost|plan|subscription|how much|per month|upgrade)\b", re.I)
_HIGH_INTENT_PATTERNS = re.compile(r"\b(sign\s*up|subscribe|buy|purchase|start\s+trial|ready\s+to\s+buy|talk\s+to\s+sales|demo)\b", re.I)
_MEMORY_PATTERNS = re.compile(
    r"\b(what'?s\s+my\s+(name|email|platform)|do\s+you\s+remember\s+my\s+(name|email|platform)|what\s+is\s+my\s+(name|email|platform)|remember\s+my\s+(name|email|platform))\b",
    re.I,
)
_PLAN_PATTERN = re.compile(
    r"\b(basic|pro|premium|enterprise)\b\s*(?:plan|tier|subscription)?",
    re.I,
)

_PROJECT_ROOT = Path(__file__).resolve().parent
_VECTORSTORE_DIR = _PROJECT_ROOT / "vectorstore"
_EMBEDDING_BACKENDS: Tuple[Tuple[str, Callable[[], object]], ...] = (
    ("huggingface", lambda: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")),
    ("openai", OpenAIEmbeddings),
)

_FIELD_PROMPTS = {
    "name": "Thanks for your interest! May I have your name?",
    "email": "Great, could you share the best email to reach you?",
    "platform": "Which platform are you using AutoStream on (e.g., web, iOS, Android)?",
}

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_NAME_PATTERN = re.compile(
    r"(?:my\s+name\s+is|i\s*am|i'm)\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})",
    re.I,
)
_PLATFORM_KEYWORDS = {
    "ios": "iOS",
    "iphone": "iOS",
    "android": "Android",
    "web": "Web",
    "browser": "Web",
    "desktop": "Desktop",
    "mac": "Desktop",
    "windows": "Desktop",
}
_NAME_STOPWORDS = {
    "ready",
    "buy",
    "purchase",
    "purchasing",
    "subscribe",
    "subscribing",
    "signup",
    "interested",
    "looking",
    "trial",
    "demo",
}
_SHORT_REPLY_STOPWORDS = {
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "sure",
    "ok",
    "okay",
    "thanks",
    "thank",
    "thank you",
    "hello",
    "hi",
    "hey",
}


def update_user_profile_from_message(state: ConversationState) -> None:
    """Parse the latest user message for profile details and update state."""

    if not state.get("messages"):
        return

    latest = state["messages"][-1]
    if not isinstance(latest, dict):
        return

    content = latest.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return

    user_info = state.setdefault(
        "user_info",
        {"name": "", "email": "", "platform": "", "target_plan": ""},
    )
    lowered = content.lower()

    if not user_info.get("email"):
        email_match = _EMAIL_PATTERN.search(content)
        if email_match:
            user_info["email"] = email_match.group(0).strip()

    if not user_info.get("name"):
        name_match = _NAME_PATTERN.search(content)
        if name_match:
            candidate = name_match.group(1).strip()
            candidate = re.split(r"\b(?:and|&)\b", candidate, maxsplit=1)[0].strip()
            tokens = [token for token in re.split(r"[\s-]+", candidate) if token]
            if tokens and all(token.lower() not in _NAME_STOPWORDS for token in tokens):
                user_info["name"] = " ".join(word.capitalize() for word in tokens)
        else:
            compact = content.strip()
            if 1 <= len(compact) <= 40:
                candidate_tokens = [tok for tok in re.split(r"[\s-]+", compact) if tok]
                if (
                    1 <= len(candidate_tokens) <= 3
                    and all(token.isalpha() for token in candidate_tokens)
                    and all(token.lower() not in _SHORT_REPLY_STOPWORDS for token in candidate_tokens)
                ):
                    user_info["name"] = " ".join(word.capitalize() for word in candidate_tokens)

    if not user_info.get("platform"):
        for keyword, label in _PLATFORM_KEYWORDS.items():
            if keyword in lowered:
                user_info["platform"] = label
                break

    if not user_info.get("target_plan"):
        plan_match = _PLAN_PATTERN.search(content)
        if plan_match:
            user_info["target_plan"] = plan_match.group(1).strip().lower()


def classify_intent_node(state: ConversationState) -> IntentRoute:
    """Classify the latest user message and update state with the detected intent."""

    if not state["messages"]:
        raise ValueError("Conversation state contains no messages to classify")

    latest = state["messages"][-1]
    text = latest.get("content", "")

    if _GREETING_PATTERNS.search(text):
        intent: IntentRoute = "greeting"
    elif _MEMORY_PATTERNS.search(text):
        intent = "memory_check"
    elif any(
        keyword in text.lower()
        for keyword in (
            "my name",
            "my email",
            "my platform",
            "remember me",
            "what did i",
            "tell me everything",
            "what do you remember",
            "what do you know about me",
            "what information do you have",
        )
    ):
        intent = "memory_check"
    elif "which plan" in text.lower() and state.get("user_info", {}).get("target_plan"):
        intent = "memory_check"
    elif _HIGH_INTENT_PATTERNS.search(text):
        intent = "high_intent_lead"
    elif state.get("intent") == "high_intent_lead":
        intent = "high_intent_lead"
    elif _PRICING_PATTERNS.search(text):
        intent = "pricing_inquiry"
    else:
        # Default to pricing inquiry so downstream RAG node can handle general product questions.
        intent = "pricing_inquiry"

    state["intent"] = intent
    return intent


def pricing_rag_node(state: ConversationState) -> ConversationState:
    """Answer pricing questions using the local FAISS knowledge base."""

    if state.get("intent") != "pricing_inquiry":
        raise ValueError("pricing_rag_node invoked for non-pricing intent")
    if not state["messages"]:
        raise ValueError("Conversation state contains no messages to ground the query")

    load_dotenv()
    query = state["messages"][-1].get("content", "").strip()
    if not query:
        raise ValueError("Latest user message is empty; cannot perform retrieval")
    if not _VECTORSTORE_DIR.exists():
        raise FileNotFoundError(f"Vector store not found at {_VECTORSTORE_DIR}")

    errors: List[str] = []
    for backend_name, factory in _EMBEDDING_BACKENDS:
        try:
            embeddings = factory()
            store = FAISS.load_local(
                str(_VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            docs = store.similarity_search(query, k=3)
            response = _format_retrieval_answer(docs)
            state["messages"].append({"role": "assistant", "content": response})
            return state
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Embedding backend '%s' failed: %s", backend_name, exc)
            errors.append(f"{backend_name}: {exc}")

    raise RuntimeError(
        "Failed to answer pricing inquiry via RAG. Encountered: " + "; ".join(errors)
    )


def _format_retrieval_answer(docs: Sequence[object]) -> str:
    """Format retrieved documents into a concise pricing summary."""

    if not docs:
        return (
            "I could not locate pricing details in the knowledge base. "
            "Please verify the information and try again."
        )

    plans: Dict[str, Dict[str, str]] = {}
    policies: Dict[str, str] = {}
    current_plan: Optional[str] = None

    for doc in docs:
        content = getattr(doc, "page_content", "")
        for raw_line in content.replace("\r", "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                heading = stripped.lstrip("# ").strip()
                if heading.lower().endswith("plan"):
                    current_plan = heading
                    plans.setdefault(current_plan, {})
                else:
                    current_plan = None
                continue

            line = stripped.lstrip("-•").strip()
            line = line.replace("**", "")
            line = re.sub(r"\[cite:[^\]]*\]", "", line, flags=re.I)
            line = re.sub(r"\s+", " ", line)
            if not line or line.lower().startswith("cite:"):
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if current_plan:
                    plans.setdefault(current_plan, {})[key] = value
                else:
                    policies[key] = value
            elif current_plan:
                plans.setdefault(current_plan, {})[line.lower()] = ""
            else:
                policies[line.lower()] = ""

    if not plans and not policies:
        return (
            "I could not locate pricing details in the knowledge base. "
            "Please verify the information and try again."
        )

    summaries: List[str] = []
    for plan_name, attributes in plans.items():
        details: List[str] = []
        price = attributes.get("price")
        limit = attributes.get("limit")
        resolution = attributes.get("resolution")
        features = attributes.get("features")

        if price:
            details.append(f"costs {price}")
        if limit:
            details.append(f"includes {limit}")
        if resolution:
            details.append(f"offers {resolution}")
        if features:
            details.append(f"features {features}")

        if details:
            if len(details) == 1:
                summary = f"{plan_name} {details[0]}."
            else:
                summary = f"{plan_name} {', '.join(details[:-1])}, and {details[-1]}."
            summaries.append(summary)

    if policies:
        policy_lines = [f"{key.capitalize()}: {value}" for key, value in policies.items() if value]
        if policy_lines:
            summaries.append("Policy notes: " + "; ".join(policy_lines) + ".")

    if not summaries:
        return (
            "I could not locate pricing details in the knowledge base. "
            "Please verify the information and try again."
        )

    return "Based on our pricing documentation: " + " ".join(summaries)


def information_gathering_node(state: ConversationState) -> ConversationState:
    """Collect lead details for high-intent users by requesting missing fields."""

    if state.get("intent") != "high_intent_lead":
        raise ValueError("information_gathering_node invoked for non high-intent lead")

    user_info = state.get("user_info", {})
    missing_fields = [field for field in ("name", "email", "platform") if not user_info.get(field, "").strip()]

    if not missing_fields:
        return state

    if "name" in missing_fields and "email" in missing_fields:
        prompt = (
            "Thanks for your interest! Could you share your name and the best email to reach you?"
        )
        state["messages"].append({"role": "assistant", "content": prompt})
        return state

    next_field = missing_fields[0]
    prompt = _FIELD_PROMPTS[next_field]
    state["messages"].append({"role": "assistant", "content": prompt})

    return state


def lead_capture_node(state: ConversationState) -> ConversationState:
    """Trigger the mock lead capture tool once all user details are present."""

    user_info = state.get("user_info", {})
    missing_field = _next_missing_user_field(user_info)
    if missing_field:
        raise ValueError(
            "mock_lead_capture guard triggered: missing required field " + missing_field
        )

    capture_result = mock_lead_capture(
        user_info.get("name", ""),
        user_info.get("email", ""),
        user_info.get("platform", ""),
    )

    acknowledgement = (
        "Thanks! I've sent your details to our team - expect a follow-up shortly. "
        f"(ref: {capture_result})"
    )
    state["intent"] = "lead_captured"
    state["messages"].append({"role": "assistant", "content": acknowledgement})
    return state


def recall_user_profile_node(state: ConversationState) -> ConversationState:
    """Answer questions about stored user profile details."""

    if not state.get("messages"):
        return state

    latest = state["messages"][-1]
    if not isinstance(latest, dict):
        return state

    content = latest.get("content", "")
    if not isinstance(content, str) or not content.strip():
        return state

    user_info = state.get("user_info", {}) or {}
    name = user_info.get("name", "").strip()
    email = user_info.get("email", "").strip()
    platform = user_info.get("platform", "").strip()

    lowered = content.lower()
    if "plan" in lowered and user_info.get("target_plan"):
        plan = user_info.get("target_plan", "").strip()
        response = f"You said you're looking at the {plan.capitalize()} plan."
    elif "email" in lowered:
        if email:
            response = f"You told me your email is {email}."
        else:
            response = "I don’t have your email yet—could you share it with me?"
    elif "platform" in lowered:
        if platform:
            response = f"You mentioned you're on {platform}."
        else:
            response = "I don’t have your platform noted—are you on web, iOS, or something else?"
    elif "name" in lowered or name:
        if name:
            response = f"You told me your name is {name}."
        else:
            response = "I don’t have your name on file yet—mind sharing it with me?"
    elif "everything" in lowered or "all the" in lowered or "all of" in lowered:
        details = []
        if name:
            details.append(f"your name is {name}")
        if email:
            details.append(f"your email is {email}")
        if platform:
            details.append(f"you're on {platform}")
        if user_info.get("target_plan"):
            details.append(f"you're interested in the {user_info['target_plan'].capitalize()} plan")
        if details:
            if len(details) == 1:
                summary = details[0]
            else:
                summary = ", ".join(details[:-1]) + f", and {details[-1]}"
            response = f"You told me earlier that {summary}."
        else:
            response = "I’m not seeing your details yet—could you remind me what you’ve shared?"
    else:
        details = []
        if name:
            details.append(f"your name is {name}")
        if email:
            details.append(f"your email is {email}")
        if platform:
            details.append(f"you're on {platform}")
        if user_info.get("target_plan"):
            details.append(f"you're interested in the {user_info['target_plan'].capitalize()} plan")
        if details:
            if len(details) == 1:
                summary = details[0]
            else:
                summary = ", ".join(details[:-1]) + f", and {details[-1]}"
            response = f"You told me earlier that {summary}."
        else:
            response = "I’m not seeing your details yet—could you remind me of what you shared?"

    state["messages"].append({"role": "assistant", "content": response})
    state["intent"] = "memory_check"
    return state


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock tool to capture lead details; returns a confirmation reference."""

    reference = f"lead-{abs(hash((name, email, platform))) % 10_000:04d}"
    logging.info("Captured lead %s | name=%s email=%s platform=%s", reference, name, email, platform)
    return reference


def _next_missing_user_field(user_info: Dict[str, str]) -> Optional[str]:
    """Return the next missing user detail required for high-intent follow-up."""

    for field in ("name", "email", "platform"):
        value = user_info.get(field, "")
        if not isinstance(value, str) or not value.strip():
            return field
    return None


def missing_user_profile_field(user_info: Dict[str, str]) -> Optional[str]:
    """Public helper for determining which user field still needs to be collected."""

    return _next_missing_user_field(user_info)
