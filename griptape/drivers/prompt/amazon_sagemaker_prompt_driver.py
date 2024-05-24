from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from attrs import Factory, define, field

from griptape.artifacts import TextArtifact, TextChunkArtifact
from griptape.utils import import_optional_dependency

from .base_multi_model_prompt_driver import BaseMultiModelPromptDriver

if TYPE_CHECKING:
    import boto3

    from griptape.utils import PromptStack


@define
class AmazonSageMakerPromptDriver(BaseMultiModelPromptDriver):
    session: boto3.Session = field(default=Factory(lambda: import_optional_dependency("boto3").Session()), kw_only=True)
    sagemaker_client: Any = field(
        default=Factory(lambda self: self.session.client("sagemaker-runtime"), takes_self=True), kw_only=True
    )
    custom_attributes: str = field(default="accept_eula=true", kw_only=True, metadata={"serializable": True})
    stream: bool = field(default=False, kw_only=True, metadata={"serializable": True})

    @stream.validator  # pyright: ignore
    def validate_stream(self, _, stream):
        if stream:
            raise ValueError("streaming is not supported")

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        payload = {
            "inputs": self.prompt_model_driver.prompt_stack_to_model_input(prompt_stack),
            "parameters": self.prompt_model_driver.prompt_stack_to_model_params(prompt_stack),
        }
        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=self.model,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes=self.custom_attributes,
        )

        decoded_body = json.loads(response["Body"].read().decode("utf8"))

        if decoded_body:
            return self.prompt_model_driver.process_output(decoded_body)
        else:
            raise Exception("model response is empty")

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[TextChunkArtifact]:
        raise NotImplementedError("streaming is not supported")
