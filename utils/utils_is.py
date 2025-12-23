from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Optional, Tuple
import math


class ValuePromiseDiscrepancyTracker:
    """
    Tracks V(s_t) and rewards to compute k-step value promise discrepancy.

    It computes discrepancy for the earliest available state:
        D = | V_{t-k} - (sum_{i=0}^{k-1} gamma^i * r_{t-k+i} + gamma^k * V_t ) |
    with terminal truncation: if done happens within the k-step window, we stop
    and do NOT bootstrap with V_t.

    Usage pattern:
        tracker.reset()
        tracker.observe(value_0)  # first state, no prev_reward
        for t in 1..:
            tracker.observe(value_t, prev_reward=r_{t-1}, prev_done=done_{t-1})
            discs = tracker.pop_ready_discrepancies()
    """

    def __init__(self, k: int, gamma: float = 0.99, metric: str = "abs"):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0,1], got {gamma}")
        if metric not in ("abs", "sq"):
            raise ValueError(f"metric must be 'abs' or 'sq', got {metric}")

        self.k = int(k)
        self.gamma = float(gamma)
        self.metric = metric

        self._values: Deque[float] = deque()   # length ~= rewards + 1
        self._rewards: Deque[float] = deque()  # reward for transition i -> i+1
        self._dones: Deque[bool] = deque()

        self._ready: Deque[float] = deque()    # computed discrepancies ready to consume

    def reset(self) -> None:
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()
        self._ready.clear()

    def observe(self, value_t: float,
                prev_reward: Optional[float] = None,
                prev_done: bool = False) -> None:
        """
        Observe current state's value V(s_t) and (optionally) previous transition reward/done.

        Args:
            value_t: V(s_t)
            prev_reward: r_{t-1} from transition (t-1 -> t). Use None for t=0.
            prev_done: done_{t-1} for transition (t-1 -> t)
        """
        if prev_reward is not None:
            self._rewards.append(float(prev_reward))
            self._dones.append(bool(prev_done))

        self._values.append(float(value_t))

        # Try to compute as many discrepancies as possible (sliding window)
        self._compute_ready()

    def pop_ready_discrepancies(self) -> List[float]:
        """Return and clear all discrepancies computed since last pop."""
        out = list(self._ready)
        self._ready.clear()
        return out

    def has_enough_history(self) -> bool:
        """Whether we have at least one discrepancy computable."""
        return (len(self._values) >= self.k + 1) and (len(self._rewards) >= self.k)

    def _compute_ready(self) -> None:
        """
        While we have enough (k rewards, k+1 values), compute discrepancy for oldest value.
        After computing one, slide window by 1: pop left one value and one reward/done.
        """
        while self.has_enough_history():
            v0 = self._values[0]
            vk = self._values[self.k]  # value at horizon end

            # Build k-step target with terminal truncation
            target = 0.0
            discount = 1.0
            bootstrap = True

            for i in range(self.k):
                r = self._rewards[i]
                d = self._dones[i]
                target += discount * r
                if d:
                    bootstrap = False
                    break
                discount *= self.gamma

            if bootstrap:
                target += discount * vk  # discount == gamma^k

            diff = v0 - target
            if self.metric == "abs":
                disc = abs(diff)
            else:  # "sq"
                disc = diff * diff

            self._ready.append(float(disc))

            # Slide by one step: remove oldest value and oldest reward/done
            self._values.popleft()
            self._rewards.popleft()
            self._dones.popleft()


class HomeostaticThreshold:
    """
    Adaptive threshold controller to maintain a target trigger rate (homeostasis).

    We update threshold theta after observing whether a trigger happened:
        theta <- theta * exp(lr * (I(trigger) - target_rate))

    Intuition:
      - If trigger too often (I=1 > target), theta increases -> harder to trigger
      - If trigger too rare (I=0 < target), theta decreases -> easier to trigger
    """

    def __init__(self,
                 init_theta: float,
                 target_rate: float = 0.01,
                 lr: float = 0.01,
                 min_theta: float = 1e-8,
                 max_theta: float = 1e8):
        if init_theta <= 0:
            raise ValueError(f"init_theta must be positive, got {init_theta}")
        if not (0.0 < target_rate < 1.0):
            raise ValueError(f"target_rate must be in (0,1), got {target_rate}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if min_theta <= 0 or max_theta <= 0 or min_theta >= max_theta:
            raise ValueError("Invalid min_theta/max_theta bounds")

        self.theta = float(init_theta)
        self.target_rate = float(target_rate)
        self.lr = float(lr)
        self.min_theta = float(min_theta)
        self.max_theta = float(max_theta)

    def reset(self, theta: Optional[float] = None) -> None:
        """Reset theta (optional)."""
        if theta is not None:
            if theta <= 0:
                raise ValueError(f"theta must be positive, got {theta}")
            self.theta = float(theta)

    def update(self, triggered: bool) -> float:
        """Update theta given trigger boolean, return new theta."""
        x = 1.0 if triggered else 0.0
        # multiplicative update for scale robustness
        self.theta *= math.exp(self.lr * (x - self.target_rate))
        # clip
        if self.theta < self.min_theta:
            self.theta = self.min_theta
        elif self.theta > self.max_theta:
            self.theta = self.max_theta
        return self.theta

    def is_triggered(self, signal: float) -> bool:
        """Check if signal exceeds current threshold."""
        return float(signal) > self.theta


@dataclass
class InformedSwitchConfig:
    k: int = 10
    gamma: float = 0.99
    explore_len: int = 20
    # threshold homeostasis
    init_theta: float = 1.0
    target_rate: float = 0.01
    theta_lr: float = 0.01
    # discrepancy metric
    metric: str = "abs"
    # start mode: 0 exploit, 1 explore
    start_mode: int = 0


class InformedSwitchController:
    """
    A finite-state controller that switches between exploit/explore modes.

    Mode semantics:
      MODE_EXPLOIT: default action sampling (e.g., normal std)
      MODE_EXPLORE: exploration sampling (e.g., inflated std)

    Switching logic:
      - Compute discrepancy signals from tracker
      - Trigger if signal > adaptive threshold (homeostasis)
      - When in exploit and triggered: enter explore for explore_len steps
      - While exploring: ignore triggers and count down; return to exploit when done

    Integration contract:
      - Call observe(value_t, prev_reward, prev_done) once per env step,
        at beginning of step t (after value_t is available).
      - Use current_mode() BEFORE sampling action for that step.
    """

    MODE_EXPLOIT = 0
    MODE_EXPLORE = 1

    def __init__(self, cfg: InformedSwitchConfig):
        if cfg.explore_len <= 0:
            raise ValueError(f"explore_len must be positive, got {cfg.explore_len}")
        if cfg.start_mode not in (0, 1):
            raise ValueError("start_mode must be 0(exploit) or 1(explore)")

        self.cfg = cfg
        self.tracker = ValuePromiseDiscrepancyTracker(
            k=cfg.k, gamma=cfg.gamma, metric=cfg.metric
        )
        self.thresh = HomeostaticThreshold(
            init_theta=cfg.init_theta,
            target_rate=cfg.target_rate,
            lr=cfg.theta_lr,
        )

        self._mode: int = int(cfg.start_mode)
        self._explore_left: int = cfg.explore_len if self._mode == self.MODE_EXPLORE else 0

        self.last_signal: Optional[float] = None
        self.last_triggered: bool = False

    def reset_episode(self) -> None:
        """Reset episode-specific state (tracker + explore countdown). Threshold keeps adapting."""
        self.tracker.reset()
        self._mode = int(self.cfg.start_mode)
        self._explore_left = self.cfg.explore_len if self._mode == self.MODE_EXPLORE else 0
        self.last_signal = None
        self.last_triggered = False

    def current_mode(self) -> int:
        return self._mode

    def is_exploring(self) -> bool:
        return self._mode == self.MODE_EXPLORE

    def theta(self) -> float:
        return float(self.thresh.theta)

    def observe(self, value_t: float,
                prev_reward: Optional[float] = None,
                prev_done: bool = False) -> int:
        """
        Update controller with current value and previous transition reward/done.
        Returns the mode to be used for the CURRENT step t.

        Note:
          Countdown of explore_len is applied per step when observe() is called.
          This means if you set mode to explore for next step, it will last exactly
          explore_len action decisions.
        """
        # 1) Apply countdown for the mode used in the previous action decision
        if self._mode == self.MODE_EXPLORE:
            self._explore_left -= 1
            if self._explore_left <= 0:
                self._mode = self.MODE_EXPLOIT
                self._explore_left = 0

        # 2) Update discrepancy tracker
        self.tracker.observe(value_t=value_t, prev_reward=prev_reward, prev_done=prev_done)

        # 3) Consume all newly available signals (can be multiple per step)
        signals = self.tracker.pop_ready_discrepancies()
        if signals:
            # Use the most recent one as "last_signal" for logging
            self.last_signal = float(signals[-1])

        # 4) Determine trigger (only when currently exploiting)
        triggered_now = False
        if self._mode == self.MODE_EXPLOIT:
            for s in signals:
                trig = self.thresh.is_triggered(s)
                self.thresh.update(trig)  # homeostatic update per signal
                if trig:
                    triggered_now = True
            if triggered_now:
                self._mode = self.MODE_EXPLORE
                self._explore_left = int(self.cfg.explore_len)

        else:
            # still update threshold (homeostasis) even in explore?
            # Minimal choice: yes, keep adapting using signals to match target rate.
            for s in signals:
                trig = self.thresh.is_triggered(s)
                self.thresh.update(trig)

        self.last_triggered = bool(triggered_now)

        # 5) If prev_done, episode ended; runner should call reset_episode() after env.reset().
        return self._mode
