

from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Iterator
from unittest import mock


PROBE_DATES = (datetime(2001, 1, 2), datetime(2033, 11, 30))


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    chat_template_kwargs: dict | None = None,
    tokenize: bool = False,
):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **(chat_template_kwargs or {}),
    )


@contextlib.contextmanager
def frozen_template_clock(moment: datetime) -> Iterator[None]:
    """Make the template helper ``strftime_now`` see a fixed current time."""
    from transformers.utils import chat_template_utils

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with mock.patch.object(chat_template_utils, "datetime", FrozenDatetime):
        yield


def assert_date_independent(
    tokenizer,
    messages: list[dict[str, str]],
    chat_template_kwargs: dict | None = None,
) -> None:
    """Fail if the rendered prompt changes with the system date."""
    rendered = []
    for moment in PROBE_DATES:
        with frozen_template_clock(moment):
            rendered.append(
                render_chat(
                    tokenizer,
                    messages,
                    add_generation_prompt=True,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
    if rendered[0] != rendered[1]:
        raise ValueError(
            "The chat template embeds the current date. Set "
            '"chat_template_kwargs" in the config, e.g. '
            '{"date_string": "26 Jul 2024"} for Llama 3.x.'
        )
