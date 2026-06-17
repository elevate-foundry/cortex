"""
SCL Evaluator — executable semantic rules.

Turns SCL from a data description language into a programming language.
Agents can write, gossip, and execute logic expressed as SCL records.

Architecture:
  Layer 0: types.py   — SCL as data
  Layer 1: delta.py   — SCL as mutation
  Layer 2: gossip.py  — SCL as protocol
  Layer 3: eval.py    — SCL as executable logic  ← this file
  Layer 4: agents rewrite their own Layer 3 rules via Layer 1 deltas

Core concepts:
  - Condition: a predicate over scope entries (e.g., complexity > 0.8)
  - Action: what to do when conditions match (emit SCL, mutate state, call fn)
  - Rule: condition → action, expressed as SCL records
  - Binding: variable substitution ($var references)
  - RuleEngine: evaluates rules against incoming SCL records

SCL rule grammar (extending the base grammar):
  @anchor → when [key: >threshold, action: verb, ...]   # conditional rule
  @anchor → define [fn: name, body: "SCL template"]      # function definition
  @anchor → emit [record: "SCL text"]                    # emit new SCL record
  @anchor → mutate [key: value]                          # state mutation (reuses delta)

Operators from scl-codex symbol table:
  ⇒  implication (condition ⇒ action)
  ∧  conjunction (all conditions must match)
  ∨  disjunction (any condition matches)
  ¬  negation
  ∀  universal quantifier
  ∃  existential quantifier
  λ  abstraction (function definition)
  ⊤  true    ⊥  false
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Callable

from .types import Anchor, Relation, Scope, SCLRecord, SCLDocument


# ---------------------------------------------------------------------------
# Conditions — predicates over scope entries
# ---------------------------------------------------------------------------

class CompareOp(str, Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    IN = "in"
    NOT_IN = "not_in"
    MATCH = "~"      # regex match
    EXISTS = "exists"


@dataclass(frozen=True)
class Condition:
    """A single predicate: key op value.

    Examples:
      Condition("complexity", CompareOp.GT, "0.8")
      Condition("tier", CompareOp.IN, "L3,L4,L5")
      Condition("status", CompareOp.EQ, "idle")
    """
    key: str
    op: CompareOp
    value: str

    def evaluate(self, scope: dict[str, str]) -> bool:
        """Evaluate this condition against a scope dictionary."""
        if self.op == CompareOp.EXISTS:
            return self.key in scope

        actual = scope.get(self.key)
        if actual is None:
            return False

        try:
            if self.op == CompareOp.EQ:
                return actual == self.value
            elif self.op == CompareOp.NEQ:
                return actual != self.value
            elif self.op in (CompareOp.GT, CompareOp.GTE, CompareOp.LT, CompareOp.LTE):
                a = float(actual)
                b = float(self.value)
                if self.op == CompareOp.GT:
                    return a > b
                elif self.op == CompareOp.GTE:
                    return a >= b
                elif self.op == CompareOp.LT:
                    return a < b
                else:
                    return a <= b
            elif self.op == CompareOp.IN:
                return actual in self.value.split(",")
            elif self.op == CompareOp.NOT_IN:
                return actual not in self.value.split(",")
            elif self.op == CompareOp.MATCH:
                return bool(re.search(self.value, actual))
        except (ValueError, re.error):
            return False

        return False

    def to_text(self) -> str:
        return f"{self.key}{self.op.value}{self.value}"

    @classmethod
    def parse(cls, text: str) -> "Condition":
        """Parse a condition string like 'complexity>0.8' or 'status==idle'."""
        for op in sorted(CompareOp, key=lambda o: -len(o.value)):
            if op == CompareOp.EXISTS:
                continue
            idx = text.find(op.value)
            if idx > 0:
                key = text[:idx].strip()
                value = text[idx + len(op.value):].strip()
                return cls(key=key, op=op, value=value)
        # No operator found → existence check
        return cls(key=text.strip(), op=CompareOp.EXISTS, value="")


# ---------------------------------------------------------------------------
# Compound conditions — ∧ (and), ∨ (or), ¬ (not)
# ---------------------------------------------------------------------------

class LogicOp(str, Enum):
    AND = "∧"
    OR = "∨"
    NOT = "¬"


@dataclass
class CompoundCondition:
    """Logical combination of conditions.

    CompoundCondition(LogicOp.AND, [cond1, cond2])  →  cond1 ∧ cond2
    CompoundCondition(LogicOp.NOT, [cond1])          →  ¬cond1
    """
    op: LogicOp
    children: list  # list of Condition or CompoundCondition

    def evaluate(self, scope: dict[str, str]) -> bool:
        results = [
            c.evaluate(scope) if isinstance(c, Condition)
            else c.evaluate(scope)
            for c in self.children
        ]
        if self.op == LogicOp.AND:
            return all(results)
        elif self.op == LogicOp.OR:
            return any(results)
        elif self.op == LogicOp.NOT:
            return not results[0] if results else True
        return False


# ---------------------------------------------------------------------------
# Actions — what happens when conditions match
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    EMIT = "emit"           # Produce a new SCL record
    MUTATE = "mutate"       # Mutate agent state (via delta)
    CALL = "call"           # Call a named function
    ESCALATE = "escalate"   # Trigger escalation (route to higher tier)
    LOG = "log"             # Emit to audit log
    SUPPRESS = "suppress"   # Suppress/block the triggering record
    CHAIN = "chain"         # Evaluate another rule by name


@dataclass
class Action:
    """An action to execute when a rule fires.

    Templates support variable substitution: $key references
    are resolved from the triggering record's scope.

    Examples:
      Action(ActionType.EMIT, template="@router → select [tier: $tier]")
      Action(ActionType.MUTATE, params={"status": "escalated"})
      Action(ActionType.CALL, target="triage")
    """
    action_type: ActionType
    template: str = ""           # SCL template with $var references
    params: dict[str, str] = field(default_factory=dict)
    target: str = ""             # For CALL/CHAIN: function/rule name

    def to_dict(self) -> dict:
        return {
            "type": self.action_type.value,
            "template": self.template,
            "params": self.params,
            "target": self.target,
        }


# ---------------------------------------------------------------------------
# Variable binding — $var substitution
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_.]*)")


def bind_variables(template: str, bindings: dict[str, str]) -> str:
    """Replace $var references in a template with bound values.

    Supports dotted paths: $input.type resolves bindings["input.type"]
    or bindings["input"]["type"] (flat dict preferred).
    """
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        return bindings.get(var_name, match.group(0))  # Leave unresolved vars as-is
    return _VAR_PATTERN.sub(replacer, template)


def extract_bindings(record: SCLRecord) -> dict[str, str]:
    """Extract variable bindings from an SCL record.

    Creates bindings for:
      $anchor  → record.anchor.name
      $verb    → record.relation.verb
      $key     → record.scope.entries[key]  (for each scope entry)
    """
    bindings: dict[str, str] = {
        "anchor": record.anchor.name,
        "verb": record.relation.verb,
        "timestamp": str(record.timestamp_ms),
        "weight": str(record.weight),
    }
    for k, v in record.scope.entries.items():
        bindings[k] = v
    return bindings


# ---------------------------------------------------------------------------
# Rule — condition ⇒ action
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    """A named rule: when conditions match, execute actions.

    Rules are themselves expressible as SCL:
      @router → when [complexity: >0.8, action: escalate, to: L5]

    Attributes:
        name: Unique rule identifier
        anchor_match: If set, rule only fires for records with this anchor
        verb_match: If set, rule only fires for records with this verb
        conditions: Predicates over scope entries
        actions: What to do when the rule fires
        priority: Higher priority rules fire first (default 0)
        enabled: Can be toggled off without removing
        once: If True, rule fires at most once then disables itself
        description: Human-readable explanation
    """
    name: str
    anchor_match: Optional[str] = None
    verb_match: Optional[str] = None
    conditions: list[Condition] = field(default_factory=list)
    compound: Optional[CompoundCondition] = None  # Alternative to flat conditions list
    actions: list[Action] = field(default_factory=list)
    priority: int = 0
    enabled: bool = True
    once: bool = False
    fired_count: int = 0
    description: str = ""

    def matches(self, record: SCLRecord) -> bool:
        """Check if this rule matches a given SCL record."""
        if not self.enabled:
            return False

        if self.anchor_match and record.anchor.name != self.anchor_match:
            return False

        if self.verb_match and record.relation.verb != self.verb_match:
            return False

        scope = record.scope.entries

        if self.compound:
            return self.compound.evaluate(scope)

        return all(c.evaluate(scope) for c in self.conditions)

    def fire(self, record: SCLRecord) -> list["RuleResult"]:
        """Execute this rule's actions against a triggering record.

        Returns list of RuleResults (emitted records, mutations, etc).
        """
        bindings = extract_bindings(record)
        results: list[RuleResult] = []

        for action in self.actions:
            result = RuleResult(
                rule_name=self.name,
                action=action,
                trigger=record,
            )

            if action.action_type == ActionType.EMIT:
                resolved = bind_variables(action.template, bindings)
                try:
                    emitted = SCLRecord.from_text(resolved)
                    result.emitted = emitted
                except Exception as e:
                    result.error = str(e)

            elif action.action_type == ActionType.MUTATE:
                resolved_params = {
                    k: bind_variables(v, bindings)
                    for k, v in action.params.items()
                }
                result.mutations = resolved_params

            elif action.action_type == ActionType.LOG:
                resolved = bind_variables(action.template or str(record.to_text()), bindings)
                result.log_message = resolved

            elif action.action_type == ActionType.ESCALATE:
                result.escalate_to = bind_variables(action.target or "", bindings)

            elif action.action_type == ActionType.CALL:
                result.call_target = action.target

            elif action.action_type == ActionType.CHAIN:
                result.chain_target = action.target

            results.append(result)

        self.fired_count += 1
        if self.once:
            self.enabled = False

        return results

    def to_scl(self) -> SCLRecord:
        """Express this rule as an SCL record."""
        entries: dict[str, str] = {}
        if self.anchor_match:
            entries["anchor"] = self.anchor_match
        if self.verb_match:
            entries["verb"] = self.verb_match
        for c in self.conditions:
            entries[c.key] = f"{c.op.value}{c.value}"
        if self.actions:
            entries["action"] = self.actions[0].action_type.value
            if self.actions[0].target:
                entries["target"] = self.actions[0].target
            if self.actions[0].template:
                entries["template"] = self.actions[0].template
        entries["priority"] = str(self.priority)
        if self.description:
            entries["desc"] = self.description

        return SCLRecord(
            anchor=Anchor(self.name),
            relation=Relation("when"),
            scope=Scope(entries=entries),
        )

    @classmethod
    def from_scl(cls, record: SCLRecord) -> "Rule":
        """Parse a rule from an SCL 'when' record.

        Expected format:
          @rule_name → when [key: >threshold, action: emit, template: "...", ...]
        """
        if record.relation.verb != "when":
            raise ValueError(f"Expected 'when' verb, got '{record.relation.verb}'")

        entries = dict(record.scope.entries)
        name = record.anchor.name
        anchor_match = entries.pop("anchor", None)
        verb_match = entries.pop("verb", None)
        priority = int(entries.pop("priority", "0"))
        description = entries.pop("desc", "")
        action_type_str = entries.pop("action", "log")
        target = entries.pop("target", "")
        template = entries.pop("template", "")

        # Remaining entries are conditions
        conditions = []
        for k, v in entries.items():
            # Check if value starts with an operator
            for op in [CompareOp.GTE, CompareOp.LTE, CompareOp.GT, CompareOp.LT,
                        CompareOp.NEQ, CompareOp.EQ, CompareOp.IN, CompareOp.NOT_IN,
                        CompareOp.MATCH]:
                if v.startswith(op.value):
                    conditions.append(Condition(k, op, v[len(op.value):]))
                    break
            else:
                # Plain value → equality check
                conditions.append(Condition(k, CompareOp.EQ, v))

        action = Action(
            action_type=ActionType(action_type_str),
            template=template,
            target=target,
        )

        return cls(
            name=name,
            anchor_match=anchor_match,
            verb_match=verb_match,
            conditions=conditions,
            actions=[action],
            priority=priority,
            description=description,
        )


@dataclass
class RuleResult:
    """The result of a rule firing."""
    rule_name: str
    action: Action
    trigger: SCLRecord
    emitted: Optional[SCLRecord] = None
    mutations: Optional[dict[str, str]] = None
    log_message: Optional[str] = None
    escalate_to: Optional[str] = None
    call_target: Optional[str] = None
    chain_target: Optional[str] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Function Registry — λ abstractions
# ---------------------------------------------------------------------------

@dataclass
class SCLFunction:
    """A named function defined in SCL.

    Functions are SCL templates with $var placeholders.
    They can be redefined at runtime via deltas (self-modification).

    Definition:
      @agent → define [fn: triage, body: "@task → classify [category: $type]"]
    Invocation:
      engine.call("triage", {"type": "code"})
      → SCLRecord: @task → classify [category: code]
    """
    name: str
    body: str                 # SCL template
    params: list[str] = field(default_factory=list)  # Expected $var names
    version: int = 1
    author: str = ""
    created_ms: int = 0

    def __post_init__(self):
        if self.created_ms == 0:
            self.created_ms = int(time.time() * 1000)
        if not self.params:
            # Auto-detect params from $var references in body
            self.params = _VAR_PATTERN.findall(self.body)

    def invoke(self, bindings: dict[str, str]) -> SCLRecord:
        """Invoke the function with variable bindings."""
        resolved = bind_variables(self.body, bindings)
        # Strip surrounding quotes if present (from SCL scope parsing)
        if len(resolved) >= 2 and resolved[0] == '"' and resolved[-1] == '"':
            resolved = resolved[1:-1]
        return SCLRecord.from_text(resolved)

    def to_scl(self) -> SCLRecord:
        """Express as SCL define record."""
        return SCLRecord(
            anchor=Anchor(self.author or "__fn__"),
            relation=Relation("define"),
            scope=Scope(entries={
                "fn": self.name,
                "body": self.body,
                "params": ",".join(self.params),
                "version": str(self.version),
            }),
        )

    @classmethod
    def from_scl(cls, record: SCLRecord) -> "SCLFunction":
        """Parse from @agent → define [fn: name, body: "template"]."""
        if record.relation.verb != "define":
            raise ValueError(f"Expected 'define' verb, got '{record.relation.verb}'")
        entries = record.scope.entries
        return cls(
            name=entries.get("fn", ""),
            body=entries.get("body", ""),
            params=entries.get("params", "").split(",") if entries.get("params") else [],
            version=int(entries.get("version", "1")),
            author=record.anchor.name,
        )


# ---------------------------------------------------------------------------
# RuleEngine — the evaluator
# ---------------------------------------------------------------------------

class RuleEngine:
    """Evaluates SCL rules against incoming records.

    The engine is itself mutable: agents can add, remove, and redefine
    rules and functions at runtime via SCL records. When combined with
    the delta/gossip layer, rules propagate across the swarm.

    Usage:
        engine = RuleEngine()

        # Define rules
        engine.add_rule(Rule(
            name="escalate_complex",
            conditions=[Condition("complexity", CompareOp.GT, "0.8")],
            actions=[Action(ActionType.ESCALATE, target="L5")],
        ))

        # Define functions
        engine.define_function(SCLFunction(
            name="triage",
            body="@task → classify [category: $type, urgency: $priority]",
        ))

        # Process incoming SCL
        results = engine.evaluate(record)

        # Self-modify: rewrite a function via SCL
        engine.process_meta(SCLRecord.from_text(
            '@self → define [fn: triage, body: "@task → classify [category: $type, urgency: high]"]'
        ))
    """

    def __init__(self):
        self._rules: dict[str, Rule] = {}
        self._functions: dict[str, SCLFunction] = {}
        self._eval_count: int = 0
        self._fire_count: int = 0
        self._meta_handlers: dict[str, Callable] = {
            "when": self._handle_when,
            "define": self._handle_define,
            "undefine": self._handle_undefine,
            "enable": self._handle_enable,
            "disable": self._handle_disable,
        }

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: Rule) -> None:
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> Optional[Rule]:
        return self._rules.pop(name, None)

    def get_rule(self, name: str) -> Optional[Rule]:
        return self._rules.get(name)

    @property
    def rules(self) -> list[Rule]:
        return sorted(self._rules.values(), key=lambda r: -r.priority)

    # ------------------------------------------------------------------
    # Function management
    # ------------------------------------------------------------------

    def define_function(self, fn: SCLFunction) -> None:
        existing = self._functions.get(fn.name)
        if existing:
            fn.version = existing.version + 1
        self._functions[fn.name] = fn

    def get_function(self, name: str) -> Optional[SCLFunction]:
        return self._functions.get(name)

    def call(self, fn_name: str, bindings: dict[str, str]) -> Optional[SCLRecord]:
        """Invoke a named function with bindings."""
        fn = self._functions.get(fn_name)
        if fn is None:
            return None
        return fn.invoke(bindings)

    @property
    def functions(self) -> dict[str, SCLFunction]:
        return dict(self._functions)

    # ------------------------------------------------------------------
    # Evaluation — the core loop
    # ------------------------------------------------------------------

    def evaluate(self, record: SCLRecord) -> list[RuleResult]:
        """Evaluate all rules against an incoming SCL record.

        Rules are evaluated in priority order. Each matching rule fires
        and produces results. Chain actions trigger recursive evaluation.

        Returns all results from all fired rules.
        """
        self._eval_count += 1
        all_results: list[RuleResult] = []

        for rule in self.rules:
            if rule.matches(record):
                results = rule.fire(record)
                self._fire_count += 1
                all_results.extend(results)

                # Handle chain actions (recursive rule evaluation)
                for result in results:
                    if result.chain_target:
                        chained_rule = self.get_rule(result.chain_target)
                        if chained_rule and chained_rule.matches(record):
                            chain_results = chained_rule.fire(record)
                            all_results.extend(chain_results)

                    # Handle call actions (function invocation)
                    if result.call_target:
                        bindings = extract_bindings(record)
                        emitted = self.call(result.call_target, bindings)
                        if emitted:
                            result.emitted = emitted

        return all_results

    def evaluate_document(self, doc: SCLDocument) -> list[RuleResult]:
        """Evaluate all rules against every record in a document."""
        all_results: list[RuleResult] = []
        for record in doc:
            all_results.extend(self.evaluate(record))
        return all_results

    # ------------------------------------------------------------------
    # Meta-evaluation — self-modification via SCL
    # ------------------------------------------------------------------

    def process_meta(self, record: SCLRecord) -> bool:
        """Process a meta-level SCL record that modifies the engine itself.

        Recognized verbs:
          when     → add/update a rule
          define   → add/update a function
          undefine → remove a function
          enable   → enable a rule
          disable  → disable a rule

        Returns True if the record was processed as meta.
        """
        handler = self._meta_handlers.get(record.relation.verb)
        if handler:
            handler(record)
            return True
        return False

    def _handle_when(self, record: SCLRecord) -> None:
        rule = Rule.from_scl(record)
        self.add_rule(rule)

    def _handle_define(self, record: SCLRecord) -> None:
        fn = SCLFunction.from_scl(record)
        self.define_function(fn)

    def _handle_undefine(self, record: SCLRecord) -> None:
        fn_name = record.scope.entries.get("fn", "")
        self._functions.pop(fn_name, None)

    def _handle_enable(self, record: SCLRecord) -> None:
        rule_name = record.scope.entries.get("rule", record.anchor.name)
        rule = self._rules.get(rule_name)
        if rule:
            rule.enabled = True

    def _handle_disable(self, record: SCLRecord) -> None:
        rule_name = record.scope.entries.get("rule", record.anchor.name)
        rule = self._rules.get(rule_name)
        if rule:
            rule.enabled = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_scl_document(self) -> SCLDocument:
        """Export all rules and functions as an SCL document."""
        records: list[SCLRecord] = []
        for rule in self.rules:
            records.append(rule.to_scl())
        for fn in self._functions.values():
            records.append(fn.to_scl())
        return SCLDocument(
            records=records,
            metadata={"type": "rule_engine", "rules": str(len(self._rules)),
                       "functions": str(len(self._functions))},
        )

    def status(self) -> dict:
        return {
            "rules": len(self._rules),
            "functions": len(self._functions),
            "evaluations": self._eval_count,
            "fires": self._fire_count,
        }

    def __repr__(self) -> str:
        return (
            f"RuleEngine({len(self._rules)} rules, "
            f"{len(self._functions)} fns, "
            f"{self._fire_count} fires)"
        )
