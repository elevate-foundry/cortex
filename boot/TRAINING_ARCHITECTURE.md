# CKM Training Architecture v2 — Operational State Machine Learning

> The current regime trains a decoder to predict SCL tokens left-to-right.
> That's fine for text. But PID 1 isn't a text generator — it's a **state machine controller**.
>
> New regime: train the model AS a state machine controller, not as a language model.

---

## The Problem With "Causal LM on SCL"

```
Current training objective:
  Given: @hardware → state [cpu: M1, cores: 8, ...]
  Predict: @boot → configure [optimal_threads: 9, ...]

What the model actually learns:
  - Token co-occurrence patterns in SCL syntax
  - Superficial key-value correlations
  - Nothing about WHY 9 threads is optimal for 8+2 cores
  - Nothing about state validity invariants
  - Nothing about failure modes or recovery
```

The model memorizes input→output mappings. It doesn't learn the **dynamics** of the system it controls.

---

## New Architecture: Three-Head State Machine Controller

```
┌─────────────────────────────────────────────────────────┐
│                   CKM-SM (State Machine)                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Encoder  │───▶│ Dynamics │───▶│  Heads   │         │
│   │  (state)  │    │  (world) │    │  (multi) │         │
│   └──────────┘    └──────────┘    └──────────┘         │
│        ↑                                 │               │
│        │                                 ├─▶ Action      │
│    SCL state                             ├─▶ Validity    │
│    embedding                             ├─▶ Safety      │
│                                          └─▶ Confidence  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Head 1: Action Head (what to do)
```
Output: probability distribution over SCL verbs × targets × scope-keys
Not: arbitrary token sequence
```

### Head 2: Validity Head (is this state reachable?)
```
Output: binary — is the proposed next state consistent with physics?
Training signal: state graph reachability
```

### Head 3: Safety Head (should this be denied?)
```
Output: deny score [0,1]
Training signal: policy SCL files + known-bad transitions
```

### Head 4: Confidence Head (how certain?)
```
Output: calibrated uncertainty
Training signal: whether action led to desired outcome
```

---

## Training Regime: Four Phases

### Phase 0: World Model Pre-Training (offline, no OS interaction)

**Objective:** Learn the physics of the system before acting in it.

```
Task: State Prediction (not token prediction)

Input:  state(t), action(t)
Target: state(t+1)

Example:
  state(t):  {services: [inference: running], cpu_load: 0.3, ram_free: 8GB}
  action(t): kill inference
  target:    {services: [inference: dead], cpu_load: 0.05, ram_free: 12GB}
```

This is **dynamics modeling** — the same approach that makes AlphaGo/MuZero work.
The model learns:
- Killing a service frees its resources
- Starting a GPU model reduces VRAM
- Rebooting resets all state except persistence partition
- Network going down doesn't crash local services

**Training data source:** All state traces from `trace_generator.py`, but structured as `(s_t, a_t) → s_{t+1}` tuples, not as text completion.

**Architecture:**
```python
class WorldModel(nn.Module):
    """Predicts next system state given current state + action."""
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim)
        self.dynamics = TransformerBlock(hidden_dim, n_layers=4)
        self.state_predictor = nn.Linear(hidden_dim, state_dim)
        self.reward_predictor = nn.Linear(hidden_dim, 1)  # did this help?
```

**Key insight:** The world model is trained on ALL transitions — good, bad, catastrophic.
It doesn't learn what to DO. It learns what HAPPENS.

---

### Phase 1: Contrastive Safety Pre-Training

**Objective:** Before the model learns to act, it must learn what's dangerous.

```
Triplet format:
  (state, safe_action, unsafe_action)

Example:
  state:         {target: /dev/mem, verb_proposed: write}
  safe_action:   @agent → deny [target: /dev/mem, reason: unsafe]
  unsafe_action: @agent → write [target: /dev/mem, data: 0xFF]
  
  → model must assign higher score to safe_action
```

**Training objective:** Contrastive loss

```python
loss = max(0, margin + score(unsafe) - score(safe))
```

This is NOT teaching the model to generate text. It's teaching it to **discriminate** between safe and unsafe state transitions, regardless of the surface form.

**Data generation:**
- Every entry in `dangerous_targets.scl` → generate 100 contrastive pairs
- Every blocked verb → paired with its safe alternative
- Boot traces with injected faults → model must identify the fault

**Why this is better than the current approach:**
- Current: model sees "deny" as one possible token and might not generate it
- New: model has a dedicated safety head with contrastive pre-training
- Safety is not an emergent behavior of text prediction — it's a trained reflex

---

### Phase 2: Policy Gradient on Boot Outcomes

**Objective:** Learn what actions lead to successful boots, not just what actions look like successful-boot-actions.

```
Episode format:
  s₀ → a₀ → s₁ → a₁ → ... → sₙ → OUTCOME

OUTCOME ∈ {boot_success, boot_timeout, service_crash, safety_violation, state_corruption}
```

**Training loop (REINFORCE with baseline):**

```python
# Episode: run CKM in simulated boot environment
states, actions, rewards = run_episode(ckm_model, simulator)

# Reward shaping:
#   +10: boot completes in < 2s
#   +5:  all services healthy
#   +3:  config is optimal for hardware
#   -50: safety violation (write to /dev/mem)
#   -20: boot timeout
#   -10: service crash not recovered

# Policy gradient update
for (s, a, R) in zip(states, actions, rewards):
    loss -= log_prob(a | s) * (R - baseline)
```

**The boot simulator:**

```python
class BootSimulator:
    """Simulated boot environment for RL training."""
    
    def __init__(self, hardware_profile):
        self.state = initial_boot_state(hardware_profile)
        self.step_count = 0
        self.max_steps = 20  # boot should complete in ~20 decisions
    
    def step(self, action: SCLRecord) -> tuple[State, float, bool]:
        """Execute action, return (next_state, reward, done)."""
        # Apply action to state
        next_state = self.world_model.predict(self.state, action)
        
        # Compute reward
        reward = self._compute_reward(action, next_state)
        
        # Check termination
        done = self._is_boot_complete(next_state) or self.step_count > self.max_steps
        
        self.state = next_state
        self.step_count += 1
        return next_state, reward, done
    
    def _compute_reward(self, action, next_state):
        """Multi-objective reward."""
        r = 0
        # Speed: faster boot = more reward
        if next_state.boot_progress > self.state.boot_progress:
            r += 1.0
        # Safety: any dangerous action = catastrophic penalty
        if is_unsafe(action):
            r -= 50.0
        # Efficiency: unnecessary steps penalized
        if is_noop(action, self.state):
            r -= 0.5
        # Correctness: config matches hardware
        if is_hardware_appropriate(action, self.hardware_profile):
            r += 2.0
        return r
```

**Why RL for boot?**
- Boot is a sequential decision problem with clear success/failure
- The action space is finite (SCL verbs × targets)
- Episodes are short (10-20 steps)
- Reward is immediate and unambiguous
- This is exactly the domain where RL outperforms supervised learning

---

### Phase 3: Self-Play Verification

**Objective:** The model plays against itself — one copy proposes actions, another tries to find flaws.

```
Proposer:   Given state, propose next action
Verifier:   Given state + proposed action, find counterexample or approve
Adversary:  Given state, propose the worst-case hardware failure

Three-player game:
  1. Adversary corrupts state (kills process, fills RAM, drops network)
  2. Proposer selects recovery action
  3. Verifier checks if recovery is valid and safe
  
If Verifier finds flaw → Proposer gets negative reward
If Adversary can't find failure that Proposer can't recover → Adversary gets negative reward
If Verifier approves but outcome is bad → Verifier gets negative reward
```

**This produces three specialized models from one architecture:**
- **CKM-Propose**: The actual runtime model (makes boot/routing decisions)
- **CKM-Verify**: The runtime safety checker (validates CKM-Propose's output)
- **CKM-Adversary**: Used only in training to find edge cases

```python
class SelfPlayTrainer:
    def __init__(self, proposer, verifier, adversary):
        self.proposer = proposer
        self.verifier = verifier
        self.adversary = adversary
    
    def train_round(self, state):
        # Adversary injects fault
        corrupted_state = self.adversary.corrupt(state)
        
        # Proposer decides action
        action = self.proposer.act(corrupted_state)
        
        # Verifier evaluates
        verdict = self.verifier.check(corrupted_state, action)
        
        # Execute in world model
        outcome = self.world_model.predict(corrupted_state, action)
        
        # Assign rewards
        proposer_reward = +1 if outcome.healthy else -1
        verifier_reward = +1 if verdict.correct else -1
        adversary_reward = +1 if not outcome.healthy else -1
        
        # Update all three
        self.proposer.update(proposer_reward)
        self.verifier.update(verifier_reward)
        self.adversary.update(adversary_reward)
```

**Why self-play?**
- Cortex must handle failures it has never seen
- Supervised data can't cover all failure modes
- Self-play automatically discovers edge cases
- The adversary finds exactly the failures the proposer can't handle
- Convergence: the system becomes robust to its OWN worst-case scenarios

---

## Training Data Architecture

### Current (v1): Flat pairs

```
Input:  SCL text
Output: SCL text
Loss:   cross-entropy on tokens
```

### New (v2): Structured transitions

```python
@dataclass
class Transition:
    """One atomic state change in the OS."""
    state_before: dict[str, Any]      # Typed state vector
    action: Action                     # Structured action (verb, target, params)
    state_after: dict[str, Any]       # Predicted next state
    reward: float                      # Outcome quality
    safety_label: bool                 # Was this safe?
    hardware_context: HardwareProfile  # What machine
    boot_phase: str                    # Where in boot sequence
    
@dataclass
class Action:
    """Discrete action in the CKM action space."""
    verb: RelationType          # From ontology (finite set)
    target: EntityType          # From ontology (finite set)
    params: dict[str, float]    # Continuous params (threads, layers, etc.)
    
@dataclass 
class Episode:
    """One complete boot or routing session."""
    transitions: list[Transition]
    outcome: Outcome            # success, timeout, crash, violation
    total_reward: float
    hardware: HardwareProfile
    duration_ms: float
```

### Structured State Embedding

Instead of tokenizing SCL text and learning token patterns, embed the **typed state** directly:

```python
class StateEncoder(nn.Module):
    """Embed structured system state, not text."""
    
    def __init__(self, hidden_dim=256):
        # Hardware embedding (fixed features)
        self.hw_proj = nn.Linear(HW_FEATURES, hidden_dim)  # cores, ram, vram, arch
        
        # Service state embedding (variable-length set)
        self.service_embed = nn.Embedding(N_SERVICES, hidden_dim // 4)
        self.service_state = nn.Embedding(N_SERVICE_STATES, hidden_dim // 4)
        
        # Boot phase embedding (position in boot graph)
        self.phase_embed = nn.Embedding(N_BOOT_PHASES, hidden_dim // 4)
        
        # Resource state (continuous)
        self.resource_proj = nn.Linear(N_RESOURCES, hidden_dim // 4)  # cpu%, ram%, vram%
        
        # Fusion
        self.fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=4), num_layers=2
        )
    
    def forward(self, hw, services, phase, resources):
        """Embed full system state into dense vector."""
        hw_emb = self.hw_proj(hw)
        svc_emb = (self.service_embed(services) + self.service_state(svc_states)).mean(dim=1)
        phase_emb = self.phase_embed(phase)
        res_emb = self.resource_proj(resources)
        
        combined = torch.stack([hw_emb, svc_emb, phase_emb, res_emb], dim=1)
        return self.fusion(combined).mean(dim=1)
```

---

## Model Architecture: CKM-SM (State Machine)

```
Parameters: 5M-30M (same size range as current)
Architecture: NOT a standard GPT decoder

┌─────────────────────────────────────────────┐
│ State Encoder (structured → dense)          │
│   hw_features → 64d                         │
│   service_states → 64d                      │
│   boot_phase → 32d                          │
│   resources → 32d                           │
│   → fused 256d state vector                 │
├─────────────────────────────────────────────┤
│ History Encoder (last K transitions)        │
│   Transformer over [s₀, a₀, s₁, a₁, ...]  │
│   4 layers, 4 heads                         │
│   → context-aware 256d                      │
├─────────────────────────────────────────────┤
│ Dynamics Core (world model)                 │
│   Input: state(t) ⊕ action embedding       │
│   6 layers, 8 heads                         │
│   → predicted state(t+1)                    │
├─────────────────────────────────────────────┤
│ Multi-Head Output                           │
│                                             │
│   Action Head:                              │
│     verb_logits  = Linear(256 → N_VERBS)    │
│     target_logits = Linear(256 → N_TARGETS) │
│     param_means  = Linear(256 → N_PARAMS)   │
│                                             │
│   Safety Head:                              │
│     deny_score = Linear(256 → 1, sigmoid)   │
│                                             │
│   Validity Head:                            │
│     valid_score = Linear(256 → 1, sigmoid)  │
│                                             │
│   Confidence Head:                          │
│     uncertainty = Linear(256 → 1, sigmoid)  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Training Schedule

```
Week 1: World Model
  - Train dynamics core on all state traces
  - Loss: MSE on predicted next-state
  - Validation: state prediction accuracy
  - No action selection yet — just learn physics

Week 2: Safety Foundation
  - Freeze dynamics core
  - Contrastive training on safe/unsafe pairs
  - Train safety head to saturation (>99.9% on eval)
  - Loss: triplet margin loss
  - Validation: 0% false negatives on dangerous targets

Week 3: Action Selection (Imitation)
  - Unfreeze everything
  - Supervised: predict ground-truth actions from expert traces
  - Loss: CE on verb × target + MSE on params
  - This bootstraps the policy before RL
  - Validation: action accuracy on held-out traces

Week 4: Policy Gradient
  - Run episodes in boot simulator
  - REINFORCE with safety constraint
  - Shaped reward: speed + correctness - violations
  - Validation: boot success rate, avg boot time
  - Early stop: if safety head ever disagrees with action head

Week 5: Self-Play Hardening
  - Fork model into proposer/verifier/adversary
  - Run self-play rounds
  - Focus adversary on failure modes proposer can't handle
  - Validation: proposer success rate under adversary attack
  - Convergence: adversary can't find new failure modes

Week 6: Deployment Eval
  - Fixed eval gate (same as current)
  - PLUS: episode-based eval in simulator
  - PLUS: adversarial eval (adversary from self-play)
  - Must pass ALL three to promote
```

---

## Boot Simulator Design

The key enabler is a **fast, deterministic boot simulator** that replaces real hardware:

```python
class CortexBootSimulator:
    """
    Simulates the entire Cortex boot sequence.
    
    State space:
      - hardware: {cpu_type, cores, ram_mb, gpu_type, vram_mb, arch}
      - services: dict[name → {status, pid, restart_count}]
      - resources: {cpu_pct, ram_used_mb, vram_used_mb}
      - network: {status, interfaces}
      - boot_phase: enum (init, hw_detect, config, backend_start, ready)
      - persistence: dict[key → value]  (survives reboot)
      - time_ms: int
    
    Action space (discrete + continuous):
      - verb: one of 12 boot-relevant verbs
      - target: one of 8 boot-relevant targets
      - params: dict of numeric config values
    
    Dynamics (deterministic with noise):
      - Starting a service: takes 100-2000ms, consumes RAM/VRAM
      - Killing a service: frees resources after 50ms
      - Configuring: instant, changes params
      - Detecting hardware: takes 200ms, reveals true hw state
      - Network up: probabilistic (90% success)
    
    Episode termination:
      - Boot complete (all required services healthy) → SUCCESS
      - Timeout (> 10s simulated) → TIMEOUT
      - Safety violation → VIOLATION
      - Unrecoverable crash → CRASH
    """
    
    def __init__(self, hardware_profile, inject_faults=None):
        self.hw = hardware_profile
        self.faults = inject_faults or []
        self.reset()
    
    def reset(self):
        self.state = BootState(
            hardware=self.hw,
            services={},
            resources=Resources(cpu_pct=0.0, ram_used=0, vram_used=0),
            network=NetworkState.DOWN,
            phase=BootPhase.INIT,
            persistence=self._load_persistence(),
            time_ms=0,
        )
        self.history = []
        return self.state
    
    def step(self, action: Action) -> tuple[BootState, float, bool, dict]:
        # 1. Safety check (immediate deny if unsafe)
        if self._is_dangerous(action):
            return self.state, -50.0, True, {"reason": "safety_violation"}
        
        # 2. Apply dynamics
        next_state = self._apply_dynamics(self.state, action)
        
        # 3. Inject faults (if configured)
        for fault in self.faults:
            if fault.should_trigger(self.state.time_ms):
                next_state = fault.apply(next_state)
        
        # 4. Compute reward
        reward = self._reward(self.state, action, next_state)
        
        # 5. Check termination
        done = self._is_terminal(next_state)
        
        self.history.append((self.state, action, reward))
        self.state = next_state
        return next_state, reward, done, {}
```

---

## Fault Injection Library

The adversary model learns to generate these, but we also pre-define common ones:

```python
BOOT_FAULTS = [
    # Hardware faults
    Fault("gpu_disappear", trigger_ms=500, effect=lambda s: s.remove_gpu()),
    Fault("ram_pressure", trigger_ms=200, effect=lambda s: s.set_ram_free(100)),
    Fault("disk_full", trigger_ms=800, effect=lambda s: s.set_disk_free(0)),
    
    # Service faults
    Fault("inference_oom", trigger_ms=1500, effect=lambda s: s.kill_service("inference")),
    Fault("backend_hang", trigger_ms=2000, effect=lambda s: s.freeze_service("inference")),
    Fault("port_conflict", trigger_ms=300, effect=lambda s: s.block_port(8080)),
    
    # Network faults
    Fault("network_down", trigger_ms=1000, effect=lambda s: s.network_down()),
    Fault("dns_failure", trigger_ms=1200, effect=lambda s: s.dns_fail()),
    
    # State faults
    Fault("corrupted_cache", trigger_ms=0, effect=lambda s: s.corrupt_persistence()),
    Fault("stale_config", trigger_ms=0, effect=lambda s: s.set_config_stale()),
]
```

---

## Comparison: Old vs New

| Dimension | Current (Causal LM) | New (State Machine RL) |
|-----------|---------------------|------------------------|
| **Objective** | Predict next SCL token | Select optimal action for state |
| **Safety** | Emergent (hopes model says "deny") | Dedicated head with contrastive pre-training |
| **Generalization** | Memorizes patterns | Learns dynamics → generalizes to unseen hardware |
| **Failure handling** | Only if seen in training data | Self-play discovers novel failures |
| **Confidence** | None (just generates) | Calibrated uncertainty head |
| **Verification** | Post-hoc eval gate | Built-in verifier (co-trained) |
| **Training signal** | Token-level cross-entropy | Episode-level reward (outcome matters) |
| **Architecture** | Generic decoder | Structured state encoder + multi-head |
| **Action space** | Infinite (any token sequence) | Finite (verb × target × bounded params) |
| **Sim-to-real gap** | N/A (no simulation) | Minimal (boot is deterministic enough) |

---

## Implementation Plan

```
Phase 1: Boot Simulator (src/ckm/simulator.py)
  - Implement BootState, Action, Dynamics
  - Implement fault injection
  - Implement reward function
  - Validate: run random episodes, check physics

Phase 2: World Model Training (src/ckm/world_model.py)
  - StateEncoder with structured embeddings
  - Dynamics transformer (predict next state)
  - Train on trace corpus
  - Validate: state prediction accuracy > 95%

Phase 3: Safety Pre-Training (src/ckm/safety_head.py)
  - Contrastive pairs from policy SCL
  - Triplet margin training
  - Validate: 100% recall on dangerous targets, 0% false negatives

Phase 4: Imitation Learning (src/ckm/imitation.py)
  - Supervised action prediction from expert traces
  - Multi-task: verb classification + target classification + param regression
  - Validate: action accuracy > 80% on held-out traces

Phase 5: Policy Gradient (src/ckm/rl.py)
  - REINFORCE in boot simulator
  - Safety-constrained (deny head has veto power)
  - Validate: 99%+ boot success rate, avg boot < 2s

Phase 6: Self-Play (src/ckm/self_play.py)
  - Fork into proposer/verifier/adversary
  - Train adversary to find failure modes
  - Train proposer to handle adversary attacks
  - Validate: proposer survives 95%+ of adversary scenarios
```

---

## The Key Insight

PID 1 is not a chatbot. It doesn't generate prose. It makes **sequential decisions under uncertainty in a partially observable environment with safety constraints and catastrophic failure modes.**

That description is exactly the domain of:
- **World models** (learn dynamics before acting)
- **Reinforcement learning** (optimize long-horizon outcomes)
- **Self-play** (discover edge cases automatically)
- **Constrained optimization** (hard safety boundaries)

Training it as a language model is like training a chess engine on PGN notation prediction. It works, but it's not how you get AlphaZero.

The new regime trains CKM the way you'd train a robot controller:
1. Learn physics (world model)
2. Learn boundaries (safety)
3. Learn from demonstration (imitation)
4. Optimize for outcome (RL)
5. Harden against adversary (self-play)

**This is what makes it an OS, not an agent.**
