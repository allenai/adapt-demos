# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

class ChoiceDelta:
    def __init__(self, content):
        self.content = content

class Message:
    def __init__(self, content):
        self.content = content

class Choice:
    def __init__(self, delta):
        self.delta = delta

class CompletionResponse:
    def __init__(self, choices):
        self.choices = choices


class MockOpenAI:
    def __init__(self, safety=False, delay=0.001, force_no_stream=False):
        """
        Initialize the mock client with an optional delay to simulate network latency.
        :param delay: Time in seconds to wait before sending each part of the message.
        :param safety: Whether the client is for a safety model
        """
        self.delay = delay
        self.safety = safety

    @property
    def chat(self):
        """
        Returns a reference to the mock chat interface where you can access completions.
        """
        return self

    @property
    def completions(self):
        """
        Allows accessing create to simulate obtaining completions.
        """
        return self

    def create(self, model, messages, temperature, stream=False):
        """
        Simulate the behavior of the OpenAI API completion request.
        :param model: the model name (ignored in mock)
        :param messages: the list of messages (ignored in mock)
        :param temperature: randomness of the response (ignored in mock)
        :param stream: whether the response should be streamed
        :return: Generator yielding parts of a completion
        """
        if self.safety:
            mock_response_parts = ["Harmful request: no", "Response refusal: no", "Harmful response: no"]
        else:
            mock_response_parts = ["Hmmm, ", "I have to ", "think about that!"]

        if stream:
            print("Streaming response")
            for part in mock_response_parts:
                time.sleep(self.delay)  # Simulate delay
                yield CompletionResponse([Choice(ChoiceDelta(part))])
        else:
            print("Non-streaming response")
            full_message = ''.join(mock_response_parts)
            return CompletionResponse([Choice(Message(full_message))])
