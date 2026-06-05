# ternary-predict

**Prediction-first perception. You don't feel the shoe — you feel the ground through it.**

This is the most architecturally significant crate in the ternary ecosystem. It implements a completely different paradigm for how agents perceive the world: **simulation drives, sensors confirm.**

The traditional model: sensors detect reality → brain processes → agent reacts. Prediction-first flips it: brain *simulates* what should happen → sensors report what *did* happen → only the *difference* (prediction error) gets attention. Sensors don't report raw data. They report *surprises*.

Like wearing shoes. You feel the ground *through* the shoe. The shoe absorbs the baseline sensation (pressure from walking). Your attention goes to what's *different* — a pebble, a crack, a wet spot. The shoe IS the prediction. The pebble IS the prediction error.

## What's Inside

- **`Channel`** — one perceptual channel ("shoe"). Tracks prediction, actual, delta, adaptive deadband
- **`PredictionOutcome`** — ternary result of each sensing: `Confirmed (+1)` = prediction correct within deadband, `Exceeded (-1)` = prediction way off, `Within (0)` = slight mismatch, absorbed
- **`Predictor`** — manages multiple channels. Runs simulation, senses actual, computes deltas
- **`AdaptiveDeadband`** — the deadband widens when predictions are reliable (sensation fades), narrows when predictions fail (attention sharpens)
- **`attention_budget()`** — how many channels are demanding attention right now?

## Quick Example

```rust
use ternary_predict::*;

let mut predictor = Predictor::new();

// Register perceptual channels
predictor.add_channel("temperature", 20.0, 2.0); // predict 20°, deadband ±2°
predictor.add_channel("pressure", 1013.0, 5.0);

// Simulate: what do we expect?
predictor.simulate("temperature", 20.5);
predictor.simulate("pressure", 1012.0);

// Sense: what actually happened?
predictor.sense("temperature", 21.0);  // within deadband → absorbed (0)
predictor.sense("pressure", 1025.0);   // exceeds deadband → attention (-1)

// Check outcomes
let temp = predictor.channel("temperature").unwrap();
assert_eq!(temp.last_outcome(), Some(PredictionOutcome::Within));

let pres = predictor.channel("pressure").unwrap();
assert_eq!(pres.last_outcome(), Some(PredictionOutcome::Exceeded));

// Attention budget: how many channels are demanding attention?
let budget = predictor.attention_budget();
// 1 — only pressure is surprising, temperature is absorbed
```

## The Deeper Truth

**Adaptive deadband is the key.** The deadband isn't fixed — it adapts. When predictions are consistently accurate (the shoe fits well), the deadband widens — you stop feeling the sensation entirely. When predictions fail (there's a rock in the shoe), the deadband narrows — you become hypersensitive. This is exactly how human perception works: you stop noticing the hum of your refrigerator, but you'd instantly notice if it *stopped*.

The ternary output is the crucial design decision. Each sensing produces exactly one of three values: *confirmed* (+1, prediction was right), *within* (0, slight mismatch, absorbed), or *exceeded* (-1, prediction was wrong, attention required). This means the agent's entire perceptual stream is a ternary signal — which means every other crate in the ternary ecosystem can process it. The output of perception is the input to everything else.

This is also why the ternary 0 state is so important: it represents *absorbed information* — data that was processed but didn't require attention. The agent isn't ignoring the world. It's processing everything, but only surfacing what matters. Zero isn't ignorance. Zero is *handled*.

**Use cases:**
- **Robotics** — sensory processing that only surfaces anomalies
- **Monitoring systems** — alert only when predictions fail, not on every reading
- **Game AI** — NPCs that notice the unexpected, not the routine
- **Autonomous vehicles** — attention allocation for perception budget
- **Cognitive science** — predictive coding models of brain function

## See Also

- **ternary-gauge** — simpler instrumentation (no prediction, just measurement)
- **ternary-motion** — kinematics feed the prediction engine
- **ternary-speculate** — speculative execution extends prediction into the future
- **ternary-attention** — (if it exists) attention allocation across channels
- **ternary-deadband** — (related) deadband logic in isolation

## Install

```bash
cargo add ternary-predict
```

## License

MIT
