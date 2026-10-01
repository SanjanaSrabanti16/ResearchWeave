from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

TRANSIENT_FAILURE_CATEGORIES = frozenset(
    {"timeout", "rate_limited", "server_error", "network_error"}
)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """The external operation is cooling down and must not be called."""


@dataclass
class _Circuit:
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class CircuitBreaker:
    """Small in-process breaker keyed by (provider, operation)."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or cooldown_seconds < 0:
            raise ValueError("Circuit-breaker limits must be positive")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self._circuits: dict[tuple[str, str], _Circuit] = {}

    def state(self, key: tuple[str, str]) -> CircuitState:
        circuit = self._circuits.get(key)
        if circuit is None:
            return CircuitState.CLOSED
        if (
            circuit.state is CircuitState.OPEN
            and circuit.opened_at is not None
            and self.clock() - circuit.opened_at >= self.cooldown_seconds
        ):
            circuit.state = CircuitState.HALF_OPEN
            circuit.probe_in_flight = False
        return circuit.state

    def before_call(self, key: tuple[str, str]) -> CircuitState:
        state = self.state(key)
        circuit = self._circuits.setdefault(key, _Circuit())
        if state is CircuitState.OPEN:
            raise CircuitOpenError(f"{key[0]} {key[1]} circuit is open")
        if state is CircuitState.HALF_OPEN:
            if circuit.probe_in_flight:
                raise CircuitOpenError(f"{key[0]} {key[1]} recovery probe is already running")
            circuit.probe_in_flight = True
        return state

    def record_success(self, key: tuple[str, str]) -> None:
        self._circuits[key] = _Circuit()

    def record_failure(
        self,
        key: tuple[str, str],
        category: Literal[
            "timeout",
            "rate_limited",
            "server_error",
            "authentication",
            "unconfigured",
            "network_error",
            "invalid_response",
            "parsing_error",
            "circuit_open",
        ]
        | str,
    ) -> CircuitState:
        circuit = self._circuits.setdefault(key, _Circuit())
        circuit.probe_in_flight = False
        if category not in TRANSIENT_FAILURE_CATEGORIES:
            return self.state(key)
        if circuit.state is CircuitState.HALF_OPEN:
            circuit.state = CircuitState.OPEN
            circuit.failures = self.failure_threshold
            circuit.opened_at = self.clock()
            return circuit.state
        circuit.failures += 1
        if circuit.failures >= self.failure_threshold:
            circuit.state = CircuitState.OPEN
            circuit.opened_at = self.clock()
        return circuit.state
