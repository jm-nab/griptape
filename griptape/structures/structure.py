from __future__ import annotations
from concurrent import futures
from graphlib import TopologicalSorter
import logging
import uuid
from abc import ABC, abstractmethod
from logging import Logger
from typing import TYPE_CHECKING, Any, Callable, Optional
from attrs import Factory, define, field
from rich.logging import RichHandler
from griptape.artifacts import BlobArtifact, TextArtifact, BaseArtifact, ErrorArtifact
from griptape.config import BaseStructureConfig, OpenAiStructureConfig, StructureConfig
from griptape.drivers import BaseEmbeddingDriver, BasePromptDriver, OpenAiEmbeddingDriver, OpenAiChatPromptDriver
from griptape.drivers.vector.local_vector_store_driver import LocalVectorStoreDriver
from griptape.engines import CsvExtractionEngine, JsonExtractionEngine, PromptSummaryEngine
from griptape.engines.rag import RagEngine
from griptape.engines.rag.modules import (
    VectorStoreRetrievalRagModule,
    RulesetsBeforeResponseRagModule,
    PromptResponseRagModule,
    MetadataBeforeResponseRagModule,
)
from griptape.engines.rag.stages import RetrievalRagStage, ResponseRagStage
from griptape.events import BaseEvent, EventListener, FinishStructureRunEvent, StartStructureRunEvent
from griptape.memory import TaskMemory
from griptape.memory.meta import MetaMemory
from griptape.memory.structure import ConversationMemory, Run
from griptape.memory.task.storage import BlobArtifactStorage, TextArtifactStorage
from griptape.utils import deprecation_warn

if TYPE_CHECKING:
    from griptape.rules import Rule, Ruleset
    from griptape.events import BaseEvent, EventListener
    from griptape.tasks import BaseTask
    from griptape.memory.structure import BaseConversationMemory


@define
class Structure(ABC):
    LOGGER_NAME = "griptape"

    id: str = field(default=Factory(lambda: uuid.uuid4().hex), kw_only=True)
    stream: Optional[bool] = field(default=None, kw_only=True)
    prompt_driver: Optional[BasePromptDriver] = field(default=None)
    embedding_driver: Optional[BaseEmbeddingDriver] = field(default=None, kw_only=True)
    config: BaseStructureConfig = field(
        default=Factory(lambda self: self.default_config, takes_self=True), kw_only=True
    )
    rulesets: list[Ruleset] = field(factory=list, kw_only=True)
    rules: list[Rule] = field(factory=list, kw_only=True)
    custom_logger: Optional[Logger] = field(default=None, kw_only=True)
    logger_level: int = field(default=logging.INFO, kw_only=True)
    event_listeners: list[EventListener] = field(factory=list, kw_only=True)
    conversation_memory: Optional[BaseConversationMemory] = field(
        default=Factory(
            lambda self: ConversationMemory(driver=self.config.conversation_memory_driver), takes_self=True
        ),
        kw_only=True,
    )
    rag_engine: RagEngine = field(default=Factory(lambda self: self.default_rag_engine, takes_self=True), kw_only=True)
    task_memory: TaskMemory = field(
        default=Factory(lambda self: self.default_task_memory, takes_self=True), kw_only=True
    )
    meta_memory: MetaMemory = field(default=Factory(lambda: MetaMemory()), kw_only=True)
    fail_fast: bool = field(default=True, kw_only=True)
    futures_executor_fn: Callable[[], futures.Executor] = field(
        default=Factory(lambda: lambda: futures.ThreadPoolExecutor()), kw_only=True
    )
    _task_graph: Optional[dict[BaseTask, set[BaseTask]]] = field(default=None, kw_only=True, alias="task_graph")
    _execution_args: tuple = ()
    _logger: Optional[Logger] = None

    def __attrs_post_init__(self) -> None:
        if self.conversation_memory:
            self.conversation_memory.structure = self

        [task.preprocess(self) for task in self.tasks]

    @rulesets.validator  # pyright: ignore
    def validate_rulesets(self, _, rulesets: list[Ruleset]) -> None:
        if not rulesets:
            return

        if self.rules:
            raise ValueError("can't have both rulesets and rules specified")

    @rules.validator  # pyright: ignore
    def validate_rules(self, _, rules: list[Rule]) -> None:
        if not rules:
            return

        if self.rulesets:
            raise ValueError("can't have both rules and rulesets specified")

    @prompt_driver.validator  # pyright: ignore
    def validate_prompt_driver(self, attribute, value):
        if value is not None:
            deprecation_warn(f"`{attribute.name}` is deprecated, use `config.prompt_driver` instead.")

    @embedding_driver.validator  # pyright: ignore
    def validate_embedding_driver(self, attribute, value):
        if value is not None:
            deprecation_warn(f"`{attribute.name}` is deprecated, use `config.embedding_driver` instead.")

    @stream.validator  # pyright: ignore
    def validate_stream(self, attribute, value):
        if value is not None:
            deprecation_warn(f"`{attribute.name}` is deprecated, use `config.prompt_driver.stream` instead.")

    @abstractmethod
    def add_task(self, task: Optional[BaseTask], **kwargs) -> Structure: ...

    @property
    @abstractmethod
    def tasks(self) -> set[BaseTask]: ...

    @property
    @abstractmethod
    def task_graph(self) -> dict[BaseTask, set[BaseTask]]: ...

    @property
    def ordered_tasks(self) -> list[BaseTask]:
        # TODO: find a way to cache this
        # should only recalculate if the graph changes
        return [self.find_task(task_id) for task_id in TopologicalSorter(self.to_graph()).static_order()]

    @property
    def execution_args(self) -> tuple:
        return self._execution_args

    @property
    def logger(self) -> Logger:
        if self.custom_logger:
            return self.custom_logger
        else:
            if self._logger is None:
                self._logger = logging.getLogger(self.LOGGER_NAME)

                self._logger.propagate = False
                self._logger.level = self.logger_level

                self._logger.handlers = [RichHandler(show_time=True, show_path=False)]
            return self._logger

    @property
    def input_task(self) -> Optional[BaseTask]:
        return self.ordered_tasks[0] if self.tasks else None

    @property
    def output_task(self) -> Optional[BaseTask]:
        return self.ordered_tasks[-1] if self.tasks else None

    @property
    def output(self) -> Optional[BaseArtifact]:
        return self.output_task.output if self.output_task is not None else None

    @property
    def finished_tasks(self) -> list[BaseTask]:
        return [s for s in self.tasks if s.is_finished()]

    @property
    def default_config(self) -> BaseStructureConfig:
        if self.prompt_driver is not None or self.embedding_driver is not None or self.stream is not None:
            config = StructureConfig()

            prompt_driver = OpenAiChatPromptDriver(model="gpt-4o") if self.prompt_driver is None else self.prompt_driver

            embedding_driver = OpenAiEmbeddingDriver() if self.embedding_driver is None else self.embedding_driver

            if self.stream is not None:
                prompt_driver.stream = self.stream

            vector_store_driver = LocalVectorStoreDriver(embedding_driver=embedding_driver)

            config.prompt_driver = prompt_driver
            config.vector_store_driver = vector_store_driver
            config.embedding_driver = embedding_driver
        else:
            config = OpenAiStructureConfig()

        return config

    @property
    def default_rag_engine(self) -> RagEngine:
        return RagEngine(
            retrieval_stage=RetrievalRagStage(
                retrieval_modules=[VectorStoreRetrievalRagModule(vector_store_driver=self.config.vector_store_driver)]
            ),
            response_stage=ResponseRagStage(
                before_response_modules=[
                    RulesetsBeforeResponseRagModule(rulesets=self.rulesets),
                    MetadataBeforeResponseRagModule(),
                ],
                response_module=PromptResponseRagModule(prompt_driver=self.config.prompt_driver),
            ),
        )

    @property
    def default_task_memory(self) -> TaskMemory:
        return TaskMemory(
            artifact_storages={
                TextArtifact: TextArtifactStorage(
                    rag_engine=self.rag_engine,
                    retrieval_rag_module_name="VectorStoreRetrievalRagModule",
                    vector_store_driver=self.config.vector_store_driver,
                    summary_engine=PromptSummaryEngine(prompt_driver=self.config.prompt_driver),
                    csv_extraction_engine=CsvExtractionEngine(prompt_driver=self.config.prompt_driver),
                    json_extraction_engine=JsonExtractionEngine(prompt_driver=self.config.prompt_driver),
                ),
                BlobArtifact: BlobArtifactStorage(),
            }
        )

    def is_finished(self) -> bool:
        return all(s.is_finished() for s in self.tasks)

    def is_executing(self) -> bool:
        return any(s for s in self.tasks if s.is_executing())

    def find_task(self, task_id: str) -> BaseTask:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task with id {task_id} doesn't exist.")

    def add_tasks(self, *tasks: BaseTask, **kwargs) -> Structure:
        [self.add_task(s) for s in tasks]
        return self

    def add_event_listener(self, event_listener: EventListener) -> EventListener:
        if event_listener not in self.event_listeners:
            self.event_listeners.append(event_listener)

        return event_listener

    def remove_event_listener(self, event_listener: EventListener) -> None:
        if event_listener in self.event_listeners:
            self.event_listeners.remove(event_listener)
        else:
            raise ValueError("Event Listener not found.")

    def publish_event(self, event: BaseEvent, flush: bool = False) -> None:
        for event_listener in self.event_listeners:
            event_listener.publish_event(event, flush)

    def context(self, task: BaseTask) -> dict[str, Any]:
        return {"args": self.execution_args, "structure": self}

    def before_run(self, args: Any) -> None:
        self._execution_args = args

        [task.reset() for task in self.tasks]
        [task.preprocess(self) for task in self.tasks]

        self.publish_event(
            StartStructureRunEvent(
                structure_id=self.id, input_task_input=self.input_task.input, input_task_output=self.input_task.output
            )
        )

    def try_run(self, *args) -> Structure:
        exit_loop = False

        while not self.is_finished() and not exit_loop:
            futures_list = {}
            ordered_tasks = self.ordered_tasks

            for task in ordered_tasks:
                if task.can_execute():
                    future = self.futures_executor_fn().submit(task.execute)
                    futures_list[future] = task

            # Wait for all tasks to complete
            for future in futures.as_completed(futures_list):
                if isinstance(future.result(), ErrorArtifact) and self.fail_fast:
                    exit_loop = True

                    break

        if self.conversation_memory and self.output is not None:
            run = Run(input=self.input_task.input, output=self.output)

            self.conversation_memory.add_run(run)

        return self

    def after_run(self) -> None:
        self.publish_event(
            FinishStructureRunEvent(
                structure_id=self.id,
                output_task_input=self.output_task.input,
                output_task_output=self.output_task.output,
            ),
            flush=True,
        )

    def run(self, *args) -> Structure:
        self.before_run(args)

        result = self.try_run(*args)

        self.after_run()

        return result

    def find_parents(self, task: Optional[BaseTask]) -> list[BaseTask]:
        if task is not None:
            for t, parents in self.task_graph.items():
                if t.id == task.id:
                    return list(parents)
        return []

    def find_children(self, task: Optional[BaseTask]) -> list[BaseTask]:
        if task is not None:
            return [n for n, p in self.task_graph.items() if task.id in {parent.id for parent in p}]
        return []

    def to_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}

        for task, parents in self.task_graph.items():
            graph[task.id] = {parent.id for parent in parents}

        return graph

    def __add__(self, other: BaseTask | list[BaseTask]) -> Structure:
        self.add_tasks(*other) if isinstance(other, list) else self + [other]
        return self
