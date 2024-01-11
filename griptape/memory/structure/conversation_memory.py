from __future__ import annotations
from attr import define
from griptape.memory.structure import Run, BaseConversationMemory
from griptape.utils import PromptStack


@define
class ConversationMemory(BaseConversationMemory):
    def try_add_run(self, run: Run) -> None:
        self.runs.append(run)

        if self.max_runs:
            while len(self.runs) > self.max_runs:
                self.runs.pop(0)

    def to_prompt_stack(self, last_n: int | None = None) -> PromptStack:
        prompt_stack = PromptStack()
        runs = self.runs[-last_n:] if last_n else self.runs
        for run in runs:
            prompt_stack.add_user_input(run.input)
            prompt_stack.add_assistant_input(run.output)
        return prompt_stack
