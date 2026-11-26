"""Render the snapshot's chat template exactly as the reference server does."""

from __future__ import annotations

from pathlib import Path

import jinja2


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


class ChatTemplate:
    def __init__(self, snap: Path, defaults: tuple[tuple[str, object], ...] = ()):
        self.defaults = dict(defaults)  # the target's template variables; a render's own kwargs win
        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            extensions=["jinja2.ext.loopcontrols"],
            undefined=jinja2.Undefined,
        )
        env.globals["raise_exception"] = _raise_exception
        env.filters["tojson"] = jinja2.filters.do_tojson if hasattr(jinja2.filters, "do_tojson") else None
        source = (snap / "chat_template.jinja").read_text()
        self._template = env.from_string(source)
        # Nemotron's templates take no reasoning_effort, only medium_effort (its "{reasoning effort: efficient}" line):
        # there a client's low or medium effort sets it, and high keeps the template's full reasoning
        self._effort_to_medium = "medium_effort" in source and "reasoning_effort" not in source

    def render(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        if self._effort_to_medium and "reasoning_effort" in kwargs:
            effort = kwargs.pop("reasoning_effort")
            kwargs.setdefault("medium_effort", effort in ("low", "medium"))
        return self._template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            **{**self.defaults, **kwargs},
        )
