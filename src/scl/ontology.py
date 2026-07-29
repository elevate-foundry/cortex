"""
SCL AI-Native OS Ontology — The complete primitive vocabulary.

This is the instruction set architecture (ISA) for the Cortex Kernel Model.

NOT "all human knowledge."
Just every atom that exists in THIS operating system:
  agents, processes, tasks, files, symbols, tools, permissions,
  memory, state transitions, failures, repairs, verifications.

The 0.6B model doesn't learn English prose about debugging.
It learns typed state transitions:

  state_before → action → observation → state_after → verification

Every event has a canonical form. The state space explodes,
but the grammar is finite and complete.

Layer stack:
  L0: types.py    — @anchor → verb [scope]
  L1: delta.py    — state mutation
  L2: gossip.py   — state propagation
  L3: eval.py     — conditional execution
  L4: ontology.py — complete operational vocabulary  ← this file
  L5: CKM model   — probabilistic scheduler/planner/interpreter
  L6: verifier    — kernel safety boundary
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Primitive Anchors — every entity type in the OS
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """Every noun in the AI-native OS."""
    # Agents & processes
    AGENT = "agent"
    PROCESS = "process"
    TASK = "task"
    GOAL = "goal"
    CONSTRAINT = "constraint"

    # Code & data
    FILE = "file"
    SYMBOL = "symbol"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    TEST = "test"
    DATABASE = "database"

    # Communication
    MESSAGE = "message"
    TOOL = "tool"
    PERMISSION = "permission"
    MEMORY = "memory"

    # Reasoning
    CLAIM = "claim"
    EVIDENCE = "evidence"
    PROOF = "proof"
    INVARIANT = "invariant"

    # State management
    ERROR = "error"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    CHECKPOINT = "checkpoint"
    MIGRATION = "migration"

    # System
    HARDWARE = "hardware"
    CONFIG = "config"
    POLICY = "policy"
    TIER = "tier"
    MODEL = "model"
    BACKEND = "backend"

    # Graphs
    EDGE = "edge"
    CYCLE = "cycle"
    DEPENDENCY = "dependency"
    GRAPH = "graph"


# ---------------------------------------------------------------------------
# Relations — every verb (transition type)
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    """Every verb in the AI-native OS."""
    # Lifecycle
    BOOT = "boot"
    SPAWN = "spawn"
    FORK = "fork"
    MERGE = "merge"
    SLEEP = "sleep"
    RESUME = "resume"
    KILL = "kill"
    HANDOFF = "handoff"

    # Observation
    OBSERVE = "observe"
    INSPECT = "inspect"
    DETECT = "detect"
    ENUMERATE = "enumerate"
    EXTRACT = "extract"
    MEASURE = "measure"

    # Data flow
    READ = "read"
    WRITE = "write"
    CALL = "call"
    IMPORT = "import"
    EMIT = "emit"
    RECEIVE = "receive"

    # Planning
    PLAN = "plan"
    PROPOSE = "propose"
    DECIDE = "decide"
    SELECT = "select"
    ROUTE = "route"
    ESCALATE = "escalate"
    SCHEDULE = "schedule"

    # Execution
    EXECUTE = "execute"
    APPLY = "apply"
    MUTATE = "mutate"
    TRANSFORM = "transform"
    COMPRESS = "compress"
    EXPAND = "expand"

    # Verification
    VERIFY = "verify"
    PROVE = "prove"
    TEST = "test"
    ASSERT = "assert_"
    CHECK = "check"
    VALIDATE = "validate"

    # Causality
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    BLOCKS = "blocks"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    INVALIDATES = "invalidates"

    # Authorization
    PERMITS = "permits"
    DENIES = "denies"
    REQUIRES = "requires"
    GRANTS = "grants"

    # Repair
    FAIL = "fail"
    DIAGNOSE = "diagnose"
    REPAIR = "repair"
    PATCH = "patch"
    ROLLBACK = "rollback"
    RECOVER = "recover"

    # Knowledge
    SUMMARIZE = "summarize"
    CLAIM = "claim"
    SUPPORT = "support"
    REFUTE = "refute"

    # State sync
    COMMIT = "commit"
    CHECKPOINT = "checkpoint"
    SNAPSHOT = "snapshot"
    SYNC = "sync"
    GOSSIP = "gossip"


# ---------------------------------------------------------------------------
# Transition types — the state machine edges
# ---------------------------------------------------------------------------

class TransitionType(str, Enum):
    """Every state transition category."""
    BOOT = "boot"
    OBSERVE = "observe"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    FAIL = "fail"
    DIAGNOSE = "diagnose"
    PATCH = "patch"
    TEST = "test"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    HANDOFF = "handoff"
    SLEEP = "sleep"
    RESUME = "resume"
    FORK = "fork"
    MERGE = "merge"


# ---------------------------------------------------------------------------
# Typed state trace — canonical form for every OS event
# ---------------------------------------------------------------------------

@dataclass
class StateTrace:
    """
    The canonical form of every event:
      state_before → action → observation → state_after → verification

    This is what the CKM model learns to predict and repair.
    """
    agent: str                          # who acted
    state_before: dict[str, str]        # SCL scope entries before
    action: str                         # verb (RelationType)
    target: str                         # what was acted on (EntityType.name)
    observation: dict[str, str]         # what was observed
    state_after: dict[str, str]         # SCL scope entries after
    verification: Optional[str] = None  # invariant check result
    tool: Optional[str] = None          # which tool/syscall was used
    permission: Optional[str] = None    # under which permission
    evidence: Optional[str] = None      # supporting evidence
    error: Optional[str] = None         # if failed

    def to_scl_trace(self) -> list[str]:
        """Render as a sequence of SCL records (the training format)."""
        lines = []

        # State before
        before_entries = ", ".join(f"{k}: {v}" for k, v in self.state_before.items())
        lines.append(f"@{self.agent} → snapshot [{before_entries}]")

        # Action
        target_entries = ", ".join(f"{k}: {v}" for k, v in self.observation.items())
        action_scope = f"target: {self.target}"
        if self.tool:
            action_scope += f", tool: {self.tool}"
        if self.permission:
            action_scope += f", permission: {self.permission}"
        lines.append(f"@{self.agent} → {self.action} [{action_scope}]")

        # Observation
        if target_entries:
            lines.append(f"@{self.target} → observe [{target_entries}]")

        # State after
        after_entries = ", ".join(f"{k}: {v}" for k, v in self.state_after.items())
        lines.append(f"@{self.agent} → mutate [{after_entries}]")

        # Verification
        if self.verification:
            lines.append(f"@verifier → check [result: {self.verification}]")
        elif self.error:
            lines.append(f"@verifier → fail [error: {self.error}]")

        return lines

    def to_scl_text(self) -> str:
        return "\n".join(self.to_scl_trace())


# ---------------------------------------------------------------------------
# CKM Task Types — what the model is trained to DO
# ---------------------------------------------------------------------------

class CKMTask(str, Enum):
    """
    The model's job is not open-ended chat.
    It predicts valid state transitions inside a typed OS.
    """
    PREDICT_NEXT_STATE = "predict_next_state"
    DETECT_INVALID_TRANSITION = "detect_invalid_transition"
    COMPLETE_MISSING_EDGE = "complete_missing_edge"
    COMPRESS_TRACE = "compress_trace"
    EXPAND_TRACE = "expand_trace"
    REPAIR_FAILED_TRACE = "repair_failed_trace"
    RANK_NEXT_ACTION = "rank_next_action"
    SELECT_TOOL = "select_tool"
    VERIFY_CLAIM = "verify_claim"
    IDENTIFY_DEAD_ABSTRACTION = "identify_dead_abstraction"
    IDENTIFY_HIDDEN_COUPLING = "identify_hidden_coupling"
    PRODUCE_MIGRATION_PLAN = "produce_migration_plan"
    DIAGNOSE_ERROR = "diagnose_error"
    PROPOSE_REPAIR = "propose_repair"


# ---------------------------------------------------------------------------
# System model — the mapping from ontology to implementation
# ---------------------------------------------------------------------------

SYSTEM_MODEL = {
    "braille_glyphs": "machine-visible encoding",
    "scl_grammar": "assembly language",
    "ontology": "instruction set",
    "ckm_model": "probabilistic scheduler/planner/interpreter",
    "verifier": "kernel safety boundary",
    "tools": "syscalls",
    "state_trace": "ABI call frame",
    "delta": "register mutation",
    "gossip": "inter-process communication",
    "fingerprint": "content-addressable identity",
}


# ---------------------------------------------------------------------------
# Ontology validation
# ---------------------------------------------------------------------------

def validate_entity(name: str) -> bool:
    """Check if an entity name is in the ontology."""
    try:
        EntityType(name)
        return True
    except ValueError:
        return False


def validate_relation(verb: str) -> bool:
    """Check if a relation verb is in the ontology."""
    try:
        RelationType(verb)
        return True
    except ValueError:
        return False


def validate_transition(transition: str) -> bool:
    """Check if a transition type is valid."""
    try:
        TransitionType(transition)
        return True
    except ValueError:
        return False


def ontology_stats() -> dict:
    """Return counts of the ontology primitives."""
    return {
        "entity_types": len(EntityType),
        "relation_types": len(RelationType),
        "transition_types": len(TransitionType),
        "ckm_tasks": len(CKMTask),
        "total_vocabulary": len(EntityType) + len(RelationType) + len(TransitionType),
    }
