from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

from attrs import define, field

from griptape.artifacts import BaseArtifact
from griptape.mixins import SerializableMixin

if TYPE_CHECKING:
    from griptape.tools import BaseTool


@define()
class ActionArtifact(BaseArtifact, SerializableMixin):
    """Represents an instance of an LLM calling a Action.

    Attributes:
        tag: The tag (unique identifier) of the action.
        name: The name (Tool name) of the action.
        path: The path (Tool activity name) of the action.
        input: The input (Tool params) of the action.
        tool: The matched Tool of the action.
        output: The output (Tool result) of the action.
    """

    @define(kw_only=True)
    class Action(SerializableMixin):
        tag: str = field(metadata={"serializable": True})
        name: str = field(metadata={"serializable": True})
        path: Optional[str] = field(default=None, metadata={"serializable": True})
        input: dict = field(factory=dict, metadata={"serializable": True})
        tool: Optional[BaseTool] = field(default=None)
        output: Optional[BaseArtifact] = field(default=None, metadata={"serializable": True})

        def __str__(self) -> str:
            return json.dumps(self.to_dict())

        def to_dict(self) -> dict:
            return {"tag": self.tag, "name": self.name, "path": self.path, "input": self.input}

    value: ActionArtifact.Action = field(metadata={"serializable": True})

    def __add__(self, other: BaseArtifact) -> ActionArtifact:
        raise NotImplementedError
