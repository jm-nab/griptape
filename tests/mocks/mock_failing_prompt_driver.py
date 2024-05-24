from collections.abc import Iterator

from attrs import define

from griptape.artifacts import TextArtifact, TextChunkArtifact
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import BaseTokenizer, OpenAiTokenizer
from griptape.utils import PromptStack


@define
class MockFailingPromptDriver(BasePromptDriver):
    max_failures: int
    current_attempt: int = 0
    model: str = "test-model"
    tokenizer: BaseTokenizer = OpenAiTokenizer(model=OpenAiTokenizer.DEFAULT_OPENAI_GPT_3_CHAT_MODEL)

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        if self.current_attempt < self.max_failures:
            self.current_attempt += 1

            raise Exception("failed attempt")
        else:
            return TextArtifact("success")

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[TextChunkArtifact]:
        if self.current_attempt < self.max_failures:
            self.current_attempt += 1

            raise Exception("failed attempt")
        else:
            yield TextChunkArtifact("success")
