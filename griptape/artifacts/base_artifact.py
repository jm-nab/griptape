from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from attrs import Factory, define, field

from griptape.mixins import SerializableMixin

if TYPE_CHECKING:
    from griptape.artifacts import BaseChunkArtifact


@define()
class BaseArtifact(SerializableMixin, ABC):
    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True, metadata={"serializable": True})
    name: str = field(
        default=Factory(lambda self: self.id, takes_self=True), kw_only=True, metadata={"serializable": True}
    )
    value: Any = field()

    @classmethod
    def value_to_bytes(cls, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        else:
            return str(value).encode()

    @classmethod
    def value_to_dict(cls, value: Any) -> dict:
        if isinstance(value, dict):
            dict_value = value
        else:
            dict_value = json.loads(value)

        return {k: v for k, v in dict_value.items()}

    def to_text(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return self.to_text()

    def __bool__(self) -> bool:
        return bool(self.value)

    def __len__(self) -> int:
        return len(self.value)

    @abstractmethod
    def __add__(self, other: BaseArtifact) -> BaseArtifact: ...

    @classmethod
    def from_chunks(cls, chunks: Sequence[BaseChunkArtifact]) -> BaseArtifact:
        raise NotImplementedError
