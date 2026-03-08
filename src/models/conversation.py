"""
Modèles pour la gestion des conversations.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(Enum):
    """Rôles possibles dans une conversation."""

    SYSTEM = "system"  # Message système
    USER = "user"  # Message utilisateur (patient)
    ASSISTANT = "assistant"  # Message assistant (infirmier/agent)

    def to_claude_format(self) -> str:

        return self.value


class Message(BaseModel):
    """Un message dans la conversation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict] = Field(default_factory=dict)

    def to_llm_format(self) -> dict:
        """Format pour l'API LLM (claude)."""
        return {"role": self.role.to_claude_format(), "content": self.content}

    def to_display_format(self) -> dict:
        """Format pour l'affichage dans l'interface."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ConversationHistory(BaseModel):
    """Historique complet d'une conversation de triage."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Métadonnées de session
    session_type: str = Field(default="interactive")  # "interactive" ou "simulation"
    is_complete: bool = Field(default=False)

    def add_message(self, role: MessageRole, content: str, metadata: Optional[dict] = None) -> None:
        """Ajoute un message à l'historique."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def add_system_message(self, content: str) -> None:
        """Ajoute un message système."""
        self.add_message(role=MessageRole.SYSTEM, content=content)

    def add_user_message(self, content: str) -> None:
        """Ajoute un message utilisateur (patient)."""
        self.add_message(role=MessageRole.USER, content=content)

    def add_assistant_message(self, content: str) -> None:
        """Ajoute un message assistant (infirmier/agent)."""
        self.add_message(role=MessageRole.ASSISTANT, content=content)

    def to_llm_format(self) -> list[dict]:
        """Convertit l'historique au format API LLM."""
        return [msg.to_llm_format() for msg in self.messages]

    def to_display_format(self) -> list[dict]:
        """Format pour affichage Streamlit."""
        return [msg.to_display_format() for msg in self.messages]

    def get_last_n_messages(self, n: int) -> list[Message]:
        """Retourne les n derniers messages."""
        return self.messages[-n:] if n <= len(self.messages) else self.messages

    def get_messages_by_role(self, role: MessageRole) -> list[Message]:
        """Retourne tous les messages d'un rôle donné."""
        return [msg for msg in self.messages if msg.role == role]

    def get_full_text(self) -> str:
        """Retourne la conversation complète en texte."""
        return "\n".join([f"{msg.role.value}: {msg.content}" for msg in self.messages])

    def get_turn_count(self) -> int:
        """Retourne le nombre de tours de dialogue."""
        user_messages = len([m for m in self.messages if m.role == MessageRole.USER])
        return user_messages

    def clear(self) -> None:
        """Vide l'historique."""
        self.messages = []
        self.updated_at = datetime.now()
        self.is_complete = False
