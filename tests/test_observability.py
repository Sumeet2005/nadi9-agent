import logging

from nadi9.observability import Timer, configure_logging, get_logger, log_event


def test_timer_measures_latency():
    with Timer() as timer:
        pass
    assert timer.duration_ms >= 0.0


def test_logger_sanitizes_secrets(caplog):
    configure_logging("INFO")
    logger = get_logger("test")

    with caplog.at_level(logging.INFO):
        log_event(
            logger,
            "api_call_started",
            run_id="run-123",
            api_key="secret-key-xyz",
            duration_ms=42.5,
        )

    assert "api_call_started" in caplog.text
    assert "run_id=run-123" in caplog.text
    assert "duration_ms=42.50" in caplog.text
    assert "secret-key-xyz" not in caplog.text
    assert "***MASKED***" in caplog.text
