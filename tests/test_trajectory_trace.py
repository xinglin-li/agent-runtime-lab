# tests/test_trajectory_trace.py
"""V2 Structured ReAct trajectory trace tests.

Verifies that every AgentStep contains the correct V2 fields
and that trace events appear in the expected causal order.
"""
import pytest

from agent_runtime.models import (
    AgentMessage,
    ToolCall,
    PlannerRationale,
    StopReason,
    AgentStep,
)
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.arithmetic import AddNumbersTool
from agent_runtime.providers.fake_provider import FakeProvider
from agent_runtime.runtime.loop import AgentRuntime


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(AddNumbersTool())
    return reg


# ── AgentStep structure tests ──────────────────────────────────

def test_direct_answer_produces_agent_step_with_stop_reason(registry):
    """A direct-answer run should produce one AgentStep with a final_answer StopReason."""
    fake_responses = [
        AgentMessage(role="assistant", content="The answer is 42."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("What is the answer?")

    assert state.status == "completed"
    assert state.stop_reason == "final_answer"

    # One AgentStep for the single direct-answer iteration.
    assert len(state.steps) == 1
    step = state.steps[0]

    assert isinstance(step, AgentStep)
    assert step.run_id == state.run_id
    assert step.step == 1
    assert step.stop_reason is not None
    assert isinstance(step.stop_reason, StopReason)
    assert step.stop_reason.reason == "final_answer"
    assert step.stop_reason.step == 1


def test_tool_call_produces_agent_step_with_rationale_and_observation(registry):
    """A tool-call run should produce AgentSteps with rationale, action, and observation."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(
                summary="Need to add 2 and 3 per user request.",
                confidence=0.95,
            ),
            tool_calls=[
                ToolCall(
                    call_id="call_001",
                    tool_name="add_numbers",
                    arguments={"a": 2, "b": 3},
                )
            ],
        ),
        AgentMessage(role="assistant", content="2 + 3 = 5."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("What is 2 + 3?")

    assert state.status == "completed"
    assert state.stop_reason == "final_answer"

    # Two steps: one for the tool call, one for the direct answer.
    assert len(state.steps) == 2

    # Step 1: tool call
    tool_step = state.steps[0]
    assert tool_step.rationale is not None
    assert tool_step.rationale.summary == "Need to add 2 and 3 per user request."
    assert tool_step.action is not None
    assert tool_step.action.call_id == "call_001"
    assert tool_step.action.tool_name == "add_numbers"
    assert tool_step.observation is not None
    assert tool_step.observation.ok is True
    assert tool_step.stop_reason is None  # Not a terminal step.

    # Step 2: direct answer
    answer_step = state.steps[1]
    assert answer_step.action is None
    assert answer_step.observation is None
    assert answer_step.stop_reason is not None
    assert answer_step.stop_reason.reason == "final_answer"


def test_max_steps_produces_stop_reason(registry):
    """max_steps_exceeded should write a StopReason into the final AgentStep."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    call_id="c1", tool_name="add_numbers",
                    arguments={"a": 1, "b": 1},
                )
            ],
        ),
        AgentMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    call_id="c2", tool_name="add_numbers",
                    arguments={"a": 1, "b": 1},
                )
            ],
        ),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
        max_steps=2,
    )
    state = runtime.run("Loop me!")

    assert state.status == "max_steps_exceeded"
    assert state.stop_reason == "max_steps_exceeded"

    # The final step should contain a StopReason.
    final_step = state.steps[-1]
    assert final_step.stop_reason is not None
    assert final_step.stop_reason.reason == "max_steps_exceeded"


# ── trace event ordering tests ─────────────────────────────────

def test_rationale_recorded_before_tool_started(registry):
    """rationale_recorded must precede tool_started in the trace timeline."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(summary="Adding numbers."),
            tool_calls=[
                ToolCall(
                    call_id="c1", tool_name="add_numbers",
                    arguments={"a": 1, "b": 2},
                )
            ],
        ),
        AgentMessage(role="assistant", content="Done."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("Add 1 + 2")

    events = state.trace_events
    rationale_idx = next(
        i for i, e in enumerate(events) if e.event_type == "rationale_recorded"
    )
    tool_started_idx = next(
        i for i, e in enumerate(events) if e.event_type == "tool_started"
    )
    assert rationale_idx < tool_started_idx, (
        f"rationale_recorded (idx={rationale_idx}) must come before "
        f"tool_started (idx={tool_started_idx})"
    )


def test_observation_recorded_after_tool_execution(registry):
    """observation_recorded must appear after tool_succeeded or tool_failed."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    call_id="c1", tool_name="add_numbers",
                    arguments={"a": 5, "b": 7},
                )
            ],
        ),
        AgentMessage(role="assistant", content="Result is 12."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("Add 5 + 7")

    events = state.trace_events
    tool_end_idx = next(
        i for i, e in enumerate(events) if e.event_type == "tool_succeeded"
    )
    obs_idx = next(
        i for i, e in enumerate(events) if e.event_type == "observation_recorded"
    )
    assert tool_end_idx < obs_idx, (
        f"tool_succeeded (idx={tool_end_idx}) must come before "
        f"observation_recorded (idx={obs_idx})"
    )


def test_stop_reason_recorded_before_run_completed(registry):
    """stop_reason_recorded must appear before run_completed in the trace."""
    fake_responses = [
        AgentMessage(role="assistant", content="Direct answer."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("Hello")

    events = state.trace_events
    stop_idx = next(
        i for i, e in enumerate(events) if e.event_type == "stop_reason_recorded"
    )
    completed_idx = next(
        i for i, e in enumerate(events) if e.event_type == "run_completed"
    )
    assert stop_idx < completed_idx, (
        f"stop_reason_recorded (idx={stop_idx}) must come before "
        f"run_completed (idx={completed_idx})"
    )


def test_tool_action_created_after_tool_call_received(registry):
    """tool_action_created must appear after tool_call_received in the trace."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    call_id="c1", tool_name="add_numbers",
                    arguments={"a": 1, "b": 1},
                )
            ],
        ),
        AgentMessage(role="assistant", content="Done."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("Add 1 + 1")

    events = state.trace_events
    received_idx = next(
        i for i, e in enumerate(events) if e.event_type == "tool_call_received"
    )
    action_idx = next(
        i for i, e in enumerate(events) if e.event_type == "tool_action_created"
    )
    assert received_idx < action_idx, (
        f"tool_call_received (idx={received_idx}) must come before "
        f"tool_action_created (idx={action_idx})"
    )


def test_no_raw_cot_in_trace(registry):
    """Trace must NOT contain raw hidden chain-of-thought — only rationale summaries."""
    fake_responses = [
        AgentMessage(
            role="assistant",
            rationale=PlannerRationale(
                summary="Simple addition requested.",
                confidence=0.9,
                cited_state_keys=["user_input"],
            ),
            tool_calls=[
                ToolCall(
                    call_id="c1", tool_name="add_numbers",
                    arguments={"a": 3, "b": 4},
                )
            ],
        ),
        AgentMessage(role="assistant", content="3 + 4 = 7."),
    ]
    runtime = AgentRuntime(
        provider=FakeProvider(fake_responses),
        tool_registry=registry,
    )
    state = runtime.run("Add 3 + 4")

    # The AgentStep rationale must be a PlannerRationale, not a raw CoT string.
    tool_step = state.steps[0]
    assert isinstance(tool_step.rationale, PlannerRationale)
    assert tool_step.rationale.summary == "Simple addition requested."

    # Trace must record rationale_recorded, not raw "thought" events.
    rationale_events = [
        e for e in state.trace_events if e.event_type == "rationale_recorded"
    ]
    assert len(rationale_events) == 1
    assert "summary" in rationale_events[0].payload