#! /usr/bin/env python

from __future__ import annotations

from typing import Dict, Set, Tuple

import attrs
import numpy as np

import pooltool.constants as const
import pooltool.ptmath as ptmath
from pooltool.events import (
    AgentType,
    Event,
    EventType,
    null_event,
    rolling_spinning_transition,
    rolling_stationary_transition,
    sliding_rolling_transition,
    spinning_stationary_transition,
)
from pooltool.events.utils import event_type_to_ball_indices
from pooltool.objects.ball.datatypes import Ball
from pooltool.system.datatypes import System


def _null() -> Dict[str, Event]:
    return {"null": null_event(time=np.inf)}


@attrs.define
class TransitionCache:
    transitions: Dict[str, Event] = attrs.field(factory=_null)

    def get_next(self) -> Event:
        return min(
            (trans for trans in self.transitions.values()), key=lambda event: event.time
        )

    def update(self, event: Event) -> None:
        """Update transition cache for all balls in Event"""
        for agent in event.agents:
            if agent.agent_type == AgentType.BALL:
                assert isinstance(ball := agent.final, Ball)
                self.transitions[agent.id] = _next_transition(ball)

    @classmethod
    def create(cls, shot: System) -> TransitionCache:
        return cls(
            {ball_id: _next_transition(ball) for ball_id, ball in shot.balls.items()}
        )


def _next_transition(ball: Ball) -> Event:
    if ball.state.s == const.stationary or ball.state.s == const.pocketed:
        return null_event(time=np.inf)

    elif ball.state.s == const.spinning:
        dtau_E = ptmath.get_spin_time(
            ball.state.rvw, ball.params.R, ball.params.u_sp, ball.params.g
        )
        return spinning_stationary_transition(ball, ball.state.t + dtau_E)

    elif ball.state.s == const.rolling:
        dtau_E_spin = ptmath.get_spin_time(
            ball.state.rvw, ball.params.R, ball.params.u_sp, ball.params.g
        )
        dtau_E_roll = ptmath.get_roll_time(
            ball.state.rvw, ball.params.u_r, ball.params.g
        )

        if dtau_E_spin > dtau_E_roll:
            return rolling_spinning_transition(ball, ball.state.t + dtau_E_roll)
        else:
            return rolling_stationary_transition(ball, ball.state.t + dtau_E_roll)

    elif ball.state.s == const.sliding:
        dtau_E = ptmath.get_slide_time(
            ball.state.rvw, ball.params.R, ball.params.u_s, ball.params.g
        )
        return sliding_rolling_transition(ball, ball.state.t + dtau_E)

    else:
        raise NotImplementedError(f"Unknown '{ball.state.s=}'")


@attrs.define
class CollisionCache:
    times: Dict[EventType, Dict[Tuple[str, str], float]] = attrs.field(factory=dict)

    @property
    def size(self) -> int:
        return sum(len(cache) for cache in self.times.values())

    def _get_invalid_ball_ids(self, event: Event) -> Set[str]:
        return {
            event.ids[ball_idx]
            for ball_idx in event_type_to_ball_indices[event.event_type]
        }

    def invalidate(self, event: Event) -> None:
        invalid_ball_ids = self._get_invalid_ball_ids(event)

        for event_type, event_times in self.times.items():
            keys_to_delete = []

            for key in event_times:
                # Identify which indices in the key should be checked based on the event type
                ball_indices = event_type_to_ball_indices.get(event_type, [])

                # Check if any of the relevant ball IDs in the key match the invalid IDs
                if any(key[idx] in invalid_ball_ids for idx in ball_indices):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del event_times[key]

    @classmethod
    def create(cls) -> CollisionCache:
        return cls()
