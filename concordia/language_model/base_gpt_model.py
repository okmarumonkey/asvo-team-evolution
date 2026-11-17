# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Collection, Sequence
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from openai import APIConnectionError, APITimeoutError, AzureOpenAI, OpenAI
from typing_extensions import override
import time
import json
from pathlib import Path
from threading import Lock

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20


class BaseGPTModel(language_model.LanguageModel):
    """Base class for GPT models (OpenAI and Azure)"""

    def __init__(
        self,
        model_name: str,
        client: AzureOpenAI | OpenAI,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
    ):
        """Initializes the base instance."""
        self._model_name = model_name
        self._measurements = measurements
        self._channel = channel
        self._client = client
        self._usage_log_path: Path | None = None
        self._usage_lock = Lock()

    def set_usage_log_path(self, path: str | Path | None) -> None:
        """Record API token usage for the current simulation trial."""
        self._usage_log_path = Path(path) if path is not None else None

    def _record_usage(self, response) -> None:
        usage = getattr(response, 'usage', None)
        if usage is None or self._usage_log_path is None:
            return
        input_details = getattr(usage, 'prompt_tokens_details', None)
        output_details = getattr(usage, 'completion_tokens_details', None)
        record = {
            'model': getattr(response, 'model', None) or self._model_name,
            'input_tokens': getattr(usage, 'prompt_tokens', 0) or 0,
            'cached_input_tokens': getattr(input_details, 'cached_tokens', 0) or 0,
            'output_tokens': getattr(usage, 'completion_tokens', 0) or 0,
            'reasoning_output_tokens': getattr(output_details, 'reasoning_tokens', 0) or 0,
            'total_tokens': getattr(usage, 'total_tokens', 0) or 0,
        }
        with self._usage_lock:
            with self._usage_log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        # Limit tokens to 4000 for GPT models
        max_tokens = min(max_tokens, 4000)

        # GPT-5, GPT-6, and o-series models use the newer completion-token parameter.
        # They also reject sampling parameters such as a non-default
        # temperature, so omit those and disable reasoning for this wrapper's
        # short, format-sensitive simulation calls.
        uses_completion_tokens = self._model_name.lower().startswith(
            ('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')
        )

        messages = [
            {'role': 'system',
             'content': ('You always continue sentences provided ' +
                         'by the user and you never repeat what ' +
                         'the user already said.')},
            {'role': 'user',
             'content': 'Question: Is Jake a turtle?\nAnswer: Jake is '},
            {'role': 'assistant',
             'content': 'not a turtle.'},
            {'role': 'user',
             'content': ('Question: What is Priya doing right now?\nAnswer: ' +
                         'Priya is currently ')},
            {'role': 'assistant',
             'content': 'sleeping.'},
            {'role': 'user',
             'content': prompt}
        ]
        #time.sleep(10)
        for attempt in range(6):
            try:
                request = dict(
                    model=self._model_name,
                    messages=messages,
                    timeout=timeout,
                )
                if uses_completion_tokens:
                    request['max_completion_tokens'] = max_tokens
                    request['reasoning_effort'] = (
                        'low' if self._model_name.lower().startswith('gpt-6.1-sol')
                        else 'none')
                else:
                    request['max_tokens'] = max_tokens
                    request['temperature'] = temperature
                    request['stop'] = terminators
                    request['seed'] = seed

                response = self._client.chat.completions.create(**request)
                self._record_usage(response)

                # ===== 新增：空响应检测 =====
                if (
                        response is None
                        or not hasattr(response, "choices")
                        or not response.choices
                        or not hasattr(response.choices[0], "message")
                        or response.choices[0].message is None
                        or not response.choices[0].message.content
                ):
                    raise ValueError("Empty response from LLM")

                # 如果一切正常，可以在这里 break 或 return
                break

            except Exception as e:
                if isinstance(e, (APIConnectionError, APITimeoutError)) and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                mess = str(e)

                # 需要重试的情况
                if (
                        "InternalServerError" in mess
                        or "503" in mess
                        or "400" in mess
                        or "Empty response" in mess
                ):
                    if attempt < 2:
                        time.sleep(5)
                        continue

                # 其它异常直接抛出
                raise

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {'raw_text_length': len(response.choices[0].message.content)},
            )

        return response.choices[0].message.content

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        prompt = (
            prompt
            + '\nRespond EXACTLY with one of the following strings:\n'
            + '\n'.join(responses) + '.'
        )

        sample = ''
        answer = ''
        for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            temperature = sampling.dynamically_adjust_temperature(
                attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS)

            sample = self.sample_text(
                prompt,
                temperature=temperature,
                seed=seed,
            )

            #time.sleep(5)

            answer = sampling.extract_choice_response(sample)
            try:
                idx = responses.index(answer)
            except ValueError:
                continue
            else:
                if self._measurements is not None:
                    self._measurements.publish_datum(
                        self._channel, {'choices_calls': attempts}
                    )
                debug = {}
                return idx, responses[idx], debug

        raise language_model.InvalidResponseError(
            (f'Too many multiple choice attempts.\nLast attempt: {sample}, ' +
             f'extracted: {answer}')
        )
