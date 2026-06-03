"""Regression tests for one-shot CLI teardown background review.

Quiet one-shot helpers (`hermes chat -Q`, `hermes_cli.oneshot`) print their
final response and then the interpreter immediately runs process teardown. A
post-response daemon background review can outlive the foreground turn and race
with provider/client shutdown, which previously surfaced as SIGABRT (`rc=-6`)
after successful artifact-writing runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import run_agent
from agent import codex_runtime
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _assistant_response(text: str = "done"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_cli_oneshot_agent(monkeypatch) -> AIAgent:
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda *args, **kwargs: _tool_defs("skill_manage"),
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())

    agent = AIAgent(
        api_key="test",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        platform="cli",
        skip_context_files=True,
        skip_memory=True,
    )
    agent.suppress_status_output = True
    agent._interruptible_api_call = MagicMock(return_value=_assistant_response("done"))
    agent._interruptible_streaming_api_call = MagicMock(return_value=_assistant_response("done"))
    agent._persist_session = lambda *args, **kwargs: None
    agent._save_trajectory = lambda *args, **kwargs: None
    agent._save_session_log = lambda *args, **kwargs: None
    return agent


def test_cli_oneshot_does_not_spawn_background_review_after_final_response(monkeypatch):
    """Machine-readable CLI one-shots must not leave daemon review work running.

    Force the skill-review cadence to trip before a successful final response.
    In parseable one-shot mode (`quiet_mode=True`, `platform="cli"`, and
    `suppress_status_output=True`) the foreground process is about to exit, so
    the best-effort review must be skipped instead of spawning a daemon thread.
    """
    agent = _make_cli_oneshot_agent(monkeypatch)
    agent._skill_nudge_interval = 10
    agent._iters_since_skill = 10
    agent._spawn_background_review = MagicMock()

    result = agent.run_conversation("finish the artifact")

    assert result["completed"] is True
    assert result["final_response"] == "done"
    agent._spawn_background_review.assert_not_called()


def test_interactive_cli_can_still_spawn_background_review(monkeypatch):
    """The one-shot guard must not disable normal interactive CLI self-review."""
    agent = _make_cli_oneshot_agent(monkeypatch)
    agent.suppress_status_output = False
    agent._skill_nudge_interval = 10
    agent._iters_since_skill = 10
    agent._spawn_background_review = MagicMock()

    result = agent.run_conversation("finish the artifact")

    assert result["completed"] is True
    agent._spawn_background_review.assert_called_once()


def test_codex_app_server_oneshot_does_not_spawn_background_review():
    """The Codex app-server path uses the same one-shot teardown guard."""
    turn = SimpleNamespace(
        final_text="done",
        interrupted=False,
        error=None,
        projected_messages=[],
        tool_iterations=10,
        should_retire=False,
        thread_id="thread-1",
        turn_id="turn-1",
    )
    agent = SimpleNamespace(
        _codex_session=SimpleNamespace(run_turn=MagicMock(return_value=turn)),
        _skill_nudge_interval=10,
        _iters_since_skill=0,
        valid_tool_names={"skill_manage"},
        _sync_external_memory_for_turn=MagicMock(),
        _spawn_background_review=MagicMock(),
        _suppress_background_review_for_current_turn=MagicMock(return_value=True),
    )

    result = codex_runtime.run_codex_app_server_turn(
        agent,
        user_message="finish the artifact",
        original_user_message="finish the artifact",
        messages=[{"role": "user", "content": "finish the artifact"}],
        effective_task_id="task-1",
        should_review_memory=False,
    )

    assert result["completed"] is True
    assert result["final_response"] == "done"
    agent._suppress_background_review_for_current_turn.assert_called_once()
    agent._spawn_background_review.assert_not_called()
