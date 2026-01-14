"""Typed structures for maintaining agent conversation context."""

from typing import Dict, List, TypedDict


class UserInfo(TypedDict):
    """Minimal user profile details."""

    name: str
    email: str
    platform: str
    target_plan: str


class ConversationState(TypedDict):
    """Conversation-level state for the agent."""

    messages: List[Dict[str, str]]
    user_info: UserInfo
    intent: str
