from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from attr import define, field
from griptape.tools import BaseTool
from griptape.memory.structure import Run
from griptape.structures import Structure
from griptape.tasks import PromptTask, ToolkitTask

if TYPE_CHECKING:
    from griptape.tasks import BaseTask


@define
class Agent(Structure):
    input_template: str = field(default=PromptTask.DEFAULT_INPUT_TEMPLATE)
    tools: list[BaseTool] = field(factory=list, kw_only=True)
    max_meta_memory_entries: Optional[int] = field(default=20, kw_only=True)

    def __attrs_post_init__(self) -> None:
        if len(self.tasks) == 0:
            if self.tools:
                task = ToolkitTask(
                    self.input_template, tools=self.tools, max_meta_memory_entries=self.max_meta_memory_entries
                )
            else:
                task = PromptTask(self.input_template, max_meta_memory_entries=self.max_meta_memory_entries)

            self.add_task(task)

        super().__attrs_post_init__()

    @property
    def task(self) -> BaseTask:
        return self.tasks[0]

    def add_task(self, task: BaseTask) -> BaseTask:
        self.tasks.clear()

        task.preprocess(self)

        self.tasks.append(task)

        return task

    def add_tasks(self, *tasks: BaseTask) -> list[BaseTask]:
        if len(tasks) > 1:
            raise ValueError("Agents can only have one task.")
        return super().add_tasks(*tasks)

    def try_run(self, *args) -> Agent:
        self._execution_args = args

        self.task.reset()

        self.task.execute()

        if self.conversation_memory:
            self.conversation_memory.add_run(self.conversation_memory.structure_to_run(self))

        self._execution_args = ()

        return self
