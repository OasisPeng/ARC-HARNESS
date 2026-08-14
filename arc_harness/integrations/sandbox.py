"""Sandbox execution providers for local ARC harness experiments."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from arc_harness.models.capabilities import ProviderDescriptor


class SandboxError(RuntimeError):
    """Base error for sandbox execution failures."""


class SandboxPolicyError(SandboxError):
    """Raised when a command violates the sandbox policy."""


@dataclass(frozen=True)
class SandboxCommand:
    """One command submitted to a sandbox provider."""

    command: Sequence[str] | str
    cwd: str | Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def argv(self) -> list[str]:
        if isinstance(self.command, str):
            return [self.command]
        return [str(part) for part in self.command]


@dataclass(frozen=True)
class SandboxResult:
    """Result captured from a sandbox command."""

    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SandboxPolicy:
    """Local subprocess safety policy.

    This is a lightweight process sandbox, not a container boundary. It is meant
    to make agent-driven local commands bounded and auditable before a stronger
    Docker/E2B-style provider is added.
    """

    timeout_seconds: float = 30.0
    max_output_chars: int = 20000
    allow_shell: bool = False
    allowed_commands: tuple[str, ...] = ()
    denied_commands: tuple[str, ...] = (
        "curl",
        "nc",
        "ncat",
        "rm",
        "scp",
        "ssh",
        "sudo",
        "wget",
    )
    allowed_cwds: tuple[str | Path, ...] = ()
    inherit_env: bool = True
    allowed_env_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive.")


class Sandbox(Protocol):
    descriptor: ProviderDescriptor

    def run(self, command: SandboxCommand | Sequence[str] | str, **kwargs: Any) -> SandboxResult:
        ...


class LocalSubprocessSandbox:
    """Bounded local subprocess runner with policy checks and captured output."""

    descriptor = ProviderDescriptor(
        capability="sandbox",
        name="local_subprocess",
        version="0.1",
        supports=("subprocess", "timeout", "output-capture", "cwd-policy", "command-policy"),
        metadata={"isolation": "process", "network": "policy-only"},
    )

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def run(self, command: SandboxCommand | Sequence[str] | str, **kwargs: Any) -> SandboxResult:
        request = self._coerce_command(command, **kwargs)
        argv = request.argv()
        shell = isinstance(request.command, str)
        cwd = self._resolve_cwd(request.cwd)
        self._check_policy(argv, shell=shell, cwd=cwd)
        env = self._build_env(request.env)
        timeout = request.timeout_seconds or self.policy.timeout_seconds
        started = time.perf_counter()

        try:
            completed = subprocess.run(
                request.command if shell else argv,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                shell=shell,
                timeout=timeout,
                check=False,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            stdout, stderr, truncated = self._truncate(completed.stdout, completed.stderr)
            return SandboxResult(
                command=argv,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                truncated=truncated,
                metadata={**request.metadata, "cwd": str(cwd) if cwd else None},
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            stdout_text = _ensure_text(exc.stdout)
            stderr_text = _ensure_text(exc.stderr)
            stdout, stderr, truncated = self._truncate(stdout_text, stderr_text)
            return SandboxResult(
                command=argv,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_ms=duration_ms,
                timed_out=True,
                truncated=truncated,
                metadata={**request.metadata, "cwd": str(cwd) if cwd else None, "timeout_seconds": timeout},
            )

    def _coerce_command(self, command: SandboxCommand | Sequence[str] | str, **kwargs: Any) -> SandboxCommand:
        if isinstance(command, SandboxCommand):
            if kwargs:
                raise TypeError("Extra keyword arguments are not allowed with SandboxCommand.")
            return command
        return SandboxCommand(command=command, **kwargs)

    def _resolve_cwd(self, cwd: str | Path | None) -> Path | None:
        if cwd is None:
            return None
        path = Path(cwd).expanduser().resolve()
        if not path.exists():
            raise SandboxPolicyError(f"Sandbox cwd does not exist: {path}")
        if not path.is_dir():
            raise SandboxPolicyError(f"Sandbox cwd is not a directory: {path}")
        return path

    def _check_policy(self, argv: list[str], *, shell: bool, cwd: Path | None) -> None:
        if not argv or not argv[0]:
            raise SandboxPolicyError("Sandbox command must be non-empty.")
        if shell and not self.policy.allow_shell:
            raise SandboxPolicyError("Shell commands are disabled by sandbox policy.")
        executable = Path(argv[0].split()[0] if shell else argv[0]).name
        if executable in self.policy.denied_commands:
            raise SandboxPolicyError(f"Command {executable!r} is denied by sandbox policy.")
        if self.policy.allowed_commands and executable not in self.policy.allowed_commands:
            raise SandboxPolicyError(f"Command {executable!r} is not allowed by sandbox policy.")
        if cwd is not None and self.policy.allowed_cwds:
            allowed = [Path(path).expanduser().resolve() for path in self.policy.allowed_cwds]
            if not any(_is_relative_to(cwd, root) for root in allowed):
                raise SandboxPolicyError(f"Sandbox cwd {cwd} is outside allowed roots.")

    def _build_env(self, extra: Mapping[str, str]) -> dict[str, str] | None:
        if not self.policy.inherit_env and not extra:
            return {}
        env = dict(os.environ) if self.policy.inherit_env else {}
        if self.policy.allowed_env_keys:
            env = {key: value for key, value in env.items() if key in self.policy.allowed_env_keys}
        for key, value in extra.items():
            if self.policy.allowed_env_keys and key not in self.policy.allowed_env_keys:
                raise SandboxPolicyError(f"Environment variable {key!r} is not allowed by sandbox policy.")
            env[str(key)] = str(value)
        return env

    def _truncate(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        limit = self.policy.max_output_chars
        total = len(stdout) + len(stderr)
        if total <= limit:
            return stdout, stderr, False
        stdout_limit = min(len(stdout), limit // 2)
        stderr_limit = max(0, limit - stdout_limit)
        return stdout[:stdout_limit], stderr[:stderr_limit], True


def _ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
