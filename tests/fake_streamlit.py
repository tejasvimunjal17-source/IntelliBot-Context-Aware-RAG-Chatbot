"""MOCK of Streamlit - FOR TESTS ONLY.

Real Streamlit could not be installed in the build environment, so this file imitates
the part of its API that app.py uses and RECORDS everything the app draws. Tests run the
real app.py against it and then inspect what was drawn (which text, inside which chat
bubble, in what order) and what the app did (which buttons/callbacks, which inputs).

IMPORTANT
* It is written from the author's knowledge of Streamlit's API. It is NOT Streamlit.
  Passing tests prove that app.py behaves correctly against THIS imitation, not that it
  looks right or works in a real browser or on a phone.
* It models the behaviours that matter for correctness: widgets return a value only on
  the run they were used (buttons, chat_input), button callbacks run BEFORE the script on
  the next run, session_state persists between runs of one session, and every browser
  session has its own state.
* Optional tests using the real streamlit.testing AppTest exist in test_app_ui.py; they are
  skipped when streamlit is not installed.
"""

import contextlib
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from unittest import mock

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class StopScript(Exception):
    """Raised by st.stop()."""


class Element:
    def __init__(self, kind: str, args: tuple, kwargs: dict, path: tuple, index: int):
        self.kind, self.args, self.kwargs, self.path, self.index = kind, args, kwargs, path, index

    @property
    def text(self) -> str:
        return str(self.args[0]) if self.args else ""

    def __repr__(self) -> str:  # helpful when a test fails
        return f"<{self.kind} {self.text[:60]!r} in {[p[0] for p in self.path]}>"


class _Context:
    """Stands in for containers, tabs, columns, expanders, chat bubbles, spinners, the sidebar."""

    _next_id = 0

    def __init__(self, fake: "FakeStreamlit", label: str, **info: Any) -> None:
        _Context._next_id += 1
        self._fake, self.label, self.info, self.id = fake, label, info, _Context._next_id

    def __enter__(self):
        self._fake._stack.append((self.label, self.id, self.info))
        if self.label == "spinner":
            self._fake.spinner_events.append(("enter", self.info.get("text", ""), len(self._fake.elements)))
            if self._fake.on_spinner:
                self._fake.on_spinner("enter")
        return self

    def __exit__(self, *exc):
        self._fake._stack.pop()
        if self.label == "spinner":
            self._fake.spinner_events.append(("exit", self.info.get("text", ""), len(self._fake.elements)))
            if self._fake.on_spinner:
                self._fake.on_spinner("exit")
        return False

    # columns and containers can be used as st.<thing> too (col.button(...), col.slider(...))
    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        recorder = getattr(self._fake, name)

        def call(*args, **kwargs):
            with self:
                return recorder(*args, **kwargs)

        return call


class FakeStreamlit:
    """One run of the script. Created by AppRunner; tests read its recorded elements."""

    def __init__(self, session_state: dict, callbacks: dict, press: Sequence[str], chat_text: Optional[str],
                 uploads: list, text_inputs: dict, checked: set, on_spinner: Optional[Callable] = None):
        self.session_state = session_state
        self.secrets: dict = {}
        self.elements: List[Element] = []
        self._stack: List[tuple] = []
        self._callbacks = callbacks
        self._press, self._chat_text = set(press), chat_text
        self._uploads, self._text_inputs, self._checked = uploads, text_inputs, checked
        self.spinner_events: List[tuple] = []
        self.on_spinner = on_spinner
        self.sidebar = _Context(self, "sidebar")
        self.page_config: dict = {}
        self.chat_input_returned: Optional[str] = None

    # ------------------------------------------------------------ recording
    def _rec(self, kind: str, args: tuple = (), kwargs: Optional[dict] = None) -> Element:
        element = Element(kind, args, kwargs or {}, tuple((l, i) for l, i, _ in self._stack), len(self.elements))
        self.elements.append(element)
        return element

    def __getattr__(self, name: str):
        """Any other st.xxx(...) call is simply recorded (so new UI calls never crash the tests)."""
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self._rec(name, args, kwargs)

        return record

    # ------------------------------------------------------------ simple elements
    def set_page_config(self, **kwargs):
        self.page_config = kwargs
        self._rec("set_page_config", (), kwargs)

    # ------------------------------------------------------------ containers
    def container(self, **kwargs):
        return _Context(self, "container", **kwargs)

    def expander(self, label, **kwargs):
        self._rec("expander", (label,), kwargs)
        return _Context(self, "expander", text=label, **kwargs)

    def spinner(self, text="", **kwargs):
        self._rec("spinner", (text,), kwargs)
        return _Context(self, "spinner", text=text)

    def chat_message(self, name, **kwargs):
        self._rec("chat_message", (name,), kwargs)
        return _Context(self, f"chat_message:{name}", role=name, **kwargs)

    def form(self, key, **kwargs):
        return _Context(self, "form", key=key)

    def tabs(self, labels):
        self._rec("tabs", (list(labels),), {})
        return [_Context(self, f"tab:{label}", text=label) for label in labels]

    def columns(self, spec, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Context(self, f"column{i}") for i in range(n)]

    # ------------------------------------------------------------ widgets
    def _pressed(self, label: str, key: Optional[str]) -> bool:
        return (key in self._press) if key else (label in self._press)

    def button(self, label, key=None, on_click=None, disabled=False, **kwargs):
        self._rec("button", (label,), {"key": key, "disabled": disabled, **kwargs})
        if on_click is not None:
            self._callbacks[key or label] = on_click  # runs before the NEXT run, like Streamlit
        return (not disabled) and self._pressed(label, key)

    def form_submit_button(self, label, **kwargs):
        self._rec("form_submit_button", (label,), kwargs)
        return self._pressed(label, kwargs.get("key"))

    def chat_input(self, placeholder="", **kwargs):
        self._rec("chat_input", (placeholder,), kwargs)
        if kwargs.get("disabled"):
            return None
        self.chat_input_returned = self._chat_text
        return self._chat_text

    def text_input(self, label, **kwargs):
        self._rec("text_input", (label,), kwargs)
        return self._text_inputs.get(label, "")

    def file_uploader(self, label, **kwargs):
        self._rec("file_uploader", (label,), kwargs)
        return list(self._uploads)

    def checkbox(self, label, **kwargs):
        self._rec("checkbox", (label,), kwargs)
        return label in self._checked

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **kwargs):
        self._rec("slider", (label,), {"value": value})
        return value

    # ------------------------------------------------------------ decorators / control
    def cache_resource(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func

    def stop(self):
        raise StopScript()

    def rerun(self):
        self._rec("rerun")

    # ------------------------------------------------------------ inspection helpers
    def all(self, kind: str) -> List[Element]:
        return [e for e in self.elements if e.kind == kind]

    def texts(self, *kinds: str) -> List[str]:
        return [e.text for e in self.elements if e.kind in kinds]

    def everything(self) -> str:
        """All text-like values drawn (for scanning for secrets or tracebacks)."""
        parts: List[str] = []
        for e in self.elements:
            parts.extend(str(a) for a in e.args)
            parts.extend(str(v) for v in e.kwargs.values())
        return "\n".join(parts)

    def bubbles(self) -> List[dict]:
        """Chat bubbles in order: [{'role': 'user'|'assistant', 'elements': [...], 'text': '...'}]."""
        order: List[str] = []
        by_id: Dict[int, dict] = {}
        for e in self.elements:
            for label, ctx_id in e.path:
                if label.startswith("chat_message:"):
                    bubble = by_id.setdefault(ctx_id, {"role": label.split(":", 1)[1], "elements": []})
                    if ctx_id not in order:
                        order.append(ctx_id)
                    bubble["elements"].append(e)
                    break
        result = []
        for ctx_id in order:
            bubble = by_id[ctx_id]
            bubble["text"] = "\n".join(
                str(a) for el in bubble["elements"] for a in el.args if el.kind != "chat_message"
            )
            result.append(bubble)
        return result

    def in_tab(self, label: str) -> List[Element]:
        return [e for e in self.elements if any(p[0] == f"tab:{label}" for p in e.path)]

    def in_sidebar(self) -> List[Element]:
        return [e for e in self.elements if any(p[0] == "sidebar" for p in e.path)]


class AppRunner:
    """One browser session: keeps its own session_state and button callbacks between runs.

    Create two AppRunners to simulate two independent visitors.
    """

    def __init__(self) -> None:
        self.state: dict = {}
        self._callbacks: dict = {}
        self.last: Optional[FakeStreamlit] = None

    def run(self, press: Sequence[str] = (), chat: Optional[str] = None, uploads: Optional[list] = None,
            text_inputs: Optional[dict] = None, checked: Sequence[str] = (),
            on_spinner: Optional[Callable] = None) -> FakeStreamlit:
        fake = FakeStreamlit(self.state, self._callbacks, press, chat, uploads or [], text_inputs or {},
                             set(checked), on_spinner)
        # Like Streamlit: callbacks of pressed buttons run first, THEN the script runs.
        for name in press:
            callback = self._callbacks.get(name)
            if callback is not None:
                _run_callback(callback, fake)
        with mock.patch.dict(sys.modules, {"streamlit": fake}):
            try:
                runpy.run_path(str(APP_PATH), run_name="__main__")
            except StopScript:
                pass
        self.last = fake
        return fake


def _run_callback(callback: Callable, fake: FakeStreamlit) -> None:
    """Callbacks in app.py use the module-level `state`; run them against this session's state.

    The callback is a closure over the previous run's module globals, whose `state` is the same
    session_state dict, so it can simply be called."""
    callback()
