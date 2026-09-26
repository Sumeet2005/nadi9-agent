import pytest

from nadi9.budget import BudgetManager
from nadi9.domain.errors import BudgetExceededError


def test_default_budget_matches_assignment_limits():
    manager = BudgetManager()

    assert manager.max_model_calls == 25
    assert manager.max_tool_calls == 50
    assert manager.model_calls_used == 0
    assert manager.tool_calls_used == 0


def test_model_call_is_consumed():
    manager = BudgetManager(max_model_calls=2)

    assert manager.can_make_model_call() is True

    manager.consume_model_call()

    assert manager.model_calls_used == 1
    assert manager.model_calls_remaining == 1


def test_tool_call_is_consumed():
    manager = BudgetManager(max_tool_calls=2)

    assert manager.can_make_tool_call() is True

    manager.consume_tool_call()

    assert manager.tool_calls_used == 1
    assert manager.tool_calls_remaining == 1


def test_model_budget_cannot_be_exceeded():
    manager = BudgetManager(max_model_calls=1)

    manager.consume_model_call()

    assert manager.can_make_model_call() is False

    with pytest.raises(BudgetExceededError, match="Model-call budget exceeded"):
        manager.consume_model_call()

    assert manager.model_calls_used == 1


def test_tool_budget_cannot_be_exceeded():
    manager = BudgetManager(max_tool_calls=1)

    manager.consume_tool_call()

    assert manager.can_make_tool_call() is False

    with pytest.raises(BudgetExceededError, match="Tool-call budget exceeded"):
        manager.consume_tool_call()

    assert manager.tool_calls_used == 1


def test_custom_budgets_are_supported():
    manager = BudgetManager(
        max_model_calls=10,
        max_tool_calls=20,
    )

    assert manager.max_model_calls == 10
    assert manager.max_tool_calls == 20


def test_invalid_model_budget_is_rejected():
    with pytest.raises(ValueError, match="max_model_calls"):
        BudgetManager(max_model_calls=0)


def test_invalid_tool_budget_is_rejected():
    with pytest.raises(ValueError, match="max_tool_calls"):
        BudgetManager(max_tool_calls=0)


def test_reset_clears_usage():
    manager = BudgetManager(
        max_model_calls=5,
        max_tool_calls=5,
    )

    manager.consume_model_call()
    manager.consume_tool_call()

    manager.reset()

    assert manager.model_calls_used == 0
    assert manager.tool_calls_used == 0
    assert manager.model_calls_remaining == 5
    assert manager.tool_calls_remaining == 5


def test_snapshot_contains_current_budget_state():
    manager = BudgetManager(
        max_model_calls=5,
        max_tool_calls=10,
    )

    manager.consume_model_call()
    manager.consume_tool_call()

    snapshot = manager.snapshot()

    assert snapshot == {
        "model_calls_used": 1,
        "tool_calls_used": 1,
        "max_model_calls": 5,
        "max_tool_calls": 10,
        "model_calls_remaining": 4,
        "tool_calls_remaining": 9,
    }