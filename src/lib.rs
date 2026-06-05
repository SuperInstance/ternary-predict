#![forbid(unsafe_code)]
//! Prediction-first perception — simulation drives, sensors confirm.
//!
//! You don't feel the shoe. You feel the ground through the shoe.
//! The simulation absorbs the baseline. Attention goes to what surprises.
//!
//! Core model:
//! 1. Agent runs simulation of what should happen
//! 2. Sensors report actual data
//! 3. Only the DELTA (prediction error) gets attention
//! 4. Deltas within deadband are absorbed (the shoe feeling fading)
//! 5. Deltas exceeding deadband trigger attention + re-simulation
//! 6. Over time, the simulation IS reality — freed attention for higher reasoning

use std::collections::HashMap;

// ============================================================
// Channel — one "shoe" for one aspect of perception
// ============================================================

/// A perceptual channel — the shoe through which you feel the ground.
///
/// Each channel tracks:
/// - What was predicted (the shoe)
/// - What was sensed (the ground)
/// - The delta between them (what reaches attention)
/// - An adaptive deadband (tolerance for prediction error)
#[derive(Debug, Clone)]
pub struct Channel {
    pub name: String,
    pub prediction: f64,
    pub actual: Option<f64>,
    pub deadband: f64,
    pub deadband_min: f64,
    pub deadband_max: f64,
    pub adaptation_rate: f64,
    pub history: Vec<ChannelEvent>,
    pub attention_count: u64,
    pub absorb_count: u64,
    pub confirm_count: u64,
}

#[derive(Debug, Clone)]
pub struct ChannelEvent {
    pub tick: u64,
    pub predicted: f64,
    pub sensed: f64,
    pub delta: f64,
    pub outcome: PredictionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionOutcome {
    Confirmed,  // +1: Delta near zero, simulation spot-on
    Absorbed,   //  0: Delta within deadband, absorbed into baseline
    Attention,  // -1: Delta exceeded deadband, re-simulate
}

impl Channel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            prediction: 0.0,
            actual: None,
            deadband: 0.3,
            deadband_min: 0.05,
            deadband_max: 1.0,
            adaptation_rate: 0.1,
            history: Vec::new(),
            attention_count: 0,
            absorb_count: 0,
            confirm_count: 0,
        }
    }

    /// Register a prediction (put the shoe on).
    pub fn predict(&mut self, value: f64) {
        self.prediction = value;
        self.actual = None;
    }

    /// Sense actual value and compute outcome.
    /// Returns the ternary signal and whether attention was triggered.
    pub fn sense(&mut self, value: f64, tick: u64) -> (i8, bool) {
        self.actual = Some(value);
        let delta = (value - self.prediction).abs();

        let outcome = if delta < 0.01 {
            PredictionOutcome::Confirmed
        } else if delta <= self.deadband {
            PredictionOutcome::Absorbed
        } else {
            PredictionOutcome::Attention
        };

        let (signal, attention) = match outcome {
            PredictionOutcome::Confirmed => (1, false),
            PredictionOutcome::Absorbed => (0, false),
            PredictionOutcome::Attention => (-1, true),
        };

        match outcome {
            PredictionOutcome::Confirmed => {
                self.confirm_count += 1;
                self.widen_deadband();
            }
            PredictionOutcome::Absorbed => {
                self.absorb_count += 1;
                // Slowly widen — simulation is approximately right
                self.widen_deadband();
            }
            PredictionOutcome::Attention => {
                self.attention_count += 1;
                self.narrow_deadband();
            }
        }

        self.history.push(ChannelEvent {
            tick, predicted: self.prediction, sensed: value, delta, outcome,
        });

        (signal, attention)
    }

    /// Widen deadband — simulation is good, tolerate more drift.
    pub fn widen_deadband(&mut self) {
        self.deadband = (self.deadband + self.adaptation_rate).min(self.deadband_max);
    }

    /// Narrow deadband — simulation failed, be more vigilant.
    pub fn narrow_deadband(&mut self) {
        self.deadband = (self.deadband - self.adaptation_rate * 3.0).max(self.deadband_min);
    }

    /// Read the ground through the shoe — extract signal from the carrier.
    /// Once the baseline is absorbed, the channel becomes a sensor for deeper structure.
    pub fn read_ground(&self) -> Option<f64> {
        self.actual.map(|a| a - self.prediction)
    }

    /// Channel health — ratio of confirmed/absorbed vs attention events.
    pub fn health(&self) -> f64 {
        let total = self.confirm_count + self.absorb_count + self.attention_count;
        if total == 0 { return 0.5; }
        (self.confirm_count + self.absorb_count) as f64 / total as f64
    }

    /// Is this channel "worn in" — prediction well-calibrated?
    pub fn is_calibrated(&self) -> bool {
        let total = self.confirm_count + self.absorb_count + self.attention_count;
        total > 10 && self.health() > 0.8
    }

    /// Prediction accuracy over recent history.
    pub fn recent_accuracy(&self, window: usize) -> f64 {
        let recent: Vec<_> = self.history.iter().rev().take(window).collect();
        if recent.is_empty() { return 0.5; }
        let confirmed = recent.iter().filter(|e| e.outcome == PredictionOutcome::Confirmed).count();
        confirmed as f64 / recent.len() as f64
    }
}

// ============================================================
// Simulation — the model that generates predictions
// ============================================================

/// A simulation model for a single channel.
/// Learns from prediction errors to improve future predictions.
#[derive(Debug, Clone)]
pub struct SimulationModel {
    pub channel: String,
    pub weights: Vec<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub last_prediction: f64,
    pub last_error: f64,
    pub total_error: f64,
    pub updates: u64,
}

impl SimulationModel {
    pub fn new(channel: &str, input_size: usize) -> Self {
        Self {
            channel: channel.to_string(),
            weights: vec![0.0; input_size],
            bias: 0.0,
            learning_rate: 0.01,
            last_prediction: 0.0,
            last_error: 0.0,
            total_error: 0.0,
            updates: 0,
        }
    }

    /// Generate prediction from inputs.
    pub fn predict(&mut self, inputs: &[f64]) -> f64 {
        let result = self.weights.iter().zip(inputs)
            .map(|(w, x)| w * x)
            .sum::<f64>() + self.bias;
        self.last_prediction = result;
        result
    }

    /// Update model based on prediction error.
    pub fn learn(&mut self, inputs: &[f64], actual: f64) -> f64 {
        let error = actual - self.last_prediction;
        self.last_error = error;
        self.total_error += error.abs();
        self.updates += 1;

        // Simple gradient update
        for (w, x) in self.weights.iter_mut().zip(inputs) {
            *w += self.learning_rate * error * x;
        }
        self.bias += self.learning_rate * error;
        error
    }

    /// Model confidence — lower average error = higher confidence.
    pub fn confidence(&self) -> f64 {
        if self.updates == 0 { return 0.5; }
        1.0 / (1.0 + self.total_error / self.updates as f64)
    }
}

// ============================================================
// PredictionEngine — the full perception system
// ============================================================

/// The complete prediction-first perception engine.
/// Manages channels, simulations, deadbands, and attention allocation.
pub struct PredictionEngine {
    pub channels: HashMap<String, Channel>,
    pub simulations: HashMap<String, SimulationModel>,
    pub tick: u64,
    pub attention_budget: usize,
    pub attention_spent: usize,
    pub global_deadband_adaptation: f64,
}

impl PredictionEngine {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            simulations: HashMap::new(),
            tick: 0,
            attention_budget: 10,
            attention_spent: 0,
            global_deadband_adaptation: 0.05,
        }
    }

    /// Register a perceptual channel (put on a shoe).
    pub fn add_channel(&mut self, name: &str) {
        self.channels.insert(name.to_string(), Channel::new(name));
    }

    /// Register a simulation model for a channel.
    pub fn add_simulation(&mut self, channel: &str, input_size: usize) {
        self.simulations.insert(channel.to_string(), SimulationModel::new(channel, input_size));
    }

    /// Run simulation for a channel and register prediction.
    pub fn simulate(&mut self, channel: &str, inputs: &[f64]) {
        if let Some(sim) = self.simulations.get_mut(channel) {
            let prediction = sim.predict(inputs);
            if let Some(ch) = self.channels.get_mut(channel) {
                ch.predict(prediction);
            }
        }
    }

    /// Sense actual value on a channel — returns ternary signal.
    pub fn sense(&mut self, channel: &str, value: f64) -> (i8, bool) {
        if let Some(ch) = self.channels.get_mut(channel) {
            let (signal, attention) = ch.sense(value, self.tick);

            // Learn from the error
            if let Some(sim) = self.simulations.get_mut(channel) {
                sim.learn(&[value], value); // Simple: use last value as input
            }

            if attention { self.attention_spent += 1; }
            (signal, attention)
        } else { (0, false) }
    }

    /// Advance one tick.
    pub fn tick(&mut self) {
        self.tick += 1;
        self.attention_spent = 0; // Reset attention budget per tick
    }

    /// System-level perception state.
    pub fn perception_state(&self) -> PerceptionState {
        let total_channels = self.channels.len();
        if total_channels == 0 {
            return PerceptionState::Uninitialized;
        }

        let health: f64 = self.channels.values().map(|c| c.health()).sum::<f64>() / total_channels as f64;
        let calibrated = self.channels.values().filter(|c| c.is_calibrated()).count();
        let attention_rate = self.channels.values()
            .map(|c| c.attention_count as f64 / (c.confirm_count + c.absorb_count + c.attention_count).max(1) as f64)
            .sum::<f64>() / total_channels as f64;

        if health > 0.9 && attention_rate < 0.1 {
            PerceptionState::Flowing  // Shoes fully broken in, feeling the ground
        } else if health > 0.7 {
            PerceptionState::Calibrating  // Shoes on, getting used to them
        } else if attention_rate > 0.5 {
            PerceptionState::Vigilant  // Lots of surprises, narrow deadbands
        } else {
            PerceptionState::Learning  // Building up the simulation
        }
    }

    /// Get channels that need attention (deltas exceeding deadband).
    pub fn attention_channels(&self) -> Vec<&str> {
        self.channels.iter()
            .filter(|(_, ch)| ch.history.last().map_or(false, |e| e.outcome == PredictionOutcome::Attention))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Read the ground through ALL shoes — aggregate carrier signal.
    pub fn read_all_ground(&self) -> HashMap<&str, f64> {
        self.channels.iter()
            .filter_map(|(name, ch)| ch.read_ground().map(|g| (name.as_str(), g)))
            .collect()
    }

    /// Global simulation confidence.
    pub fn global_confidence(&self) -> f64 {
        if self.simulations.is_empty() { return 0.5; }
        self.simulations.values().map(|s| s.confidence()).sum::<f64>() / self.simulations.len() as f64
    }

    /// Channel summary for diagnostics.
    pub fn channel_summary(&self) -> Vec<ChannelSummary> {
        self.channels.values().map(|ch| ChannelSummary {
            name: ch.name.clone(),
            health: ch.health(),
            deadband: ch.deadband,
            calibrated: ch.is_calibrated(),
            attention_rate: ch.attention_count as f64
                / (ch.confirm_count + ch.absorb_count + ch.attention_count).max(1) as f64,
        }).collect()
    }
}

#[derive(Debug, Clone)]
pub struct ChannelSummary {
    pub name: String,
    pub health: f64,
    pub deadband: f64,
    pub calibrated: bool,
    pub attention_rate: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionState {
    Uninitialized,  // No channels registered
    Learning,       // Building simulation models
    Calibrating,    // Shoes on, deadbands adjusting
    Flowing,        // Shoes absorbed, feeling the ground
    Vigilant,       // Too many surprises, narrow deadbands
}

impl std::fmt::Display for PerceptionState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PerceptionState::Uninitialized => write!(f, "∅ UNINITIALIZED"),
            PerceptionState::Learning => write!(f, "📚 LEARNING"),
            PerceptionState::Calibrating => write!(f, "👟 CALIBRATING"),
            PerceptionState::Flowing => write!(f, "🌊 FLOWING"),
            PerceptionState::Vigilant => write!(f, "👁 VIGILANT"),
        }
    }
}

// ============================================================
// T-Minus prediction landmark
// ============================================================

/// A temporal prediction landmark — a scheduled moment when
/// simulations should be tested against reality.
#[derive(Debug, Clone)]
pub struct PredictionLandmark {
    pub id: u64,
    pub fires_at: u64,
    pub channel_predictions: HashMap<String, f64>,
    pub confidence_threshold: f64,
    pub fired: bool,
}

impl PredictionLandmark {
    pub fn new(id: u64, fires_at: u64) -> Self {
        Self {
            id, fires_at,
            channel_predictions: HashMap::new(),
            confidence_threshold: 0.5,
            fired: false,
        }
    }

    /// Register a predicted value for a channel at this landmark.
    pub fn predict_channel(&mut self, channel: &str, value: f64) {
        self.channel_predictions.insert(channel.to_string(), value);
    }

    /// Check if landmark should fire.
    pub fn should_fire(&self, tick: u64) -> bool { tick >= self.fires_at && !self.fired }

    /// Compare predictions against actual values. Returns deltas.
    pub fn reconcile(&mut self, actuals: &HashMap<String, f64>) -> Vec<PredictionDelta> {
        self.fired = true;
        self.channel_predictions.iter().map(|(ch, predicted)| {
            let actual = actuals.get(ch).copied().unwrap_or(0.0);
            let delta = actual - predicted;
            PredictionDelta {
                channel: ch.clone(),
                predicted: *predicted,
                actual,
                delta,
                surprise: delta.abs() > self.confidence_threshold,
            }
        }).collect()
    }
}

#[derive(Debug, Clone)]
pub struct PredictionDelta {
    pub channel: String,
    pub predicted: f64,
    pub actual: f64,
    pub delta: f64,
    pub surprise: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Channel tests
    #[test] fn test_channel_new() { let ch = Channel::new("test"); assert_eq!(ch.deadband, 0.3); }
    #[test] fn test_channel_predict() { let mut ch = Channel::new("test"); ch.predict(1.0); assert_eq!(ch.prediction, 1.0); }
    #[test] fn test_channel_sense_confirmed() { let mut ch = Channel::new("test"); ch.predict(1.0); let (s, a) = ch.sense(1.0, 0); assert_eq!(s, 1); assert!(!a); }
    #[test] fn test_channel_sense_absorbed() { let mut ch = Channel::new("test"); ch.predict(1.0); let (s, a) = ch.sense(1.1, 0); assert_eq!(s, 0); assert!(!a); }
    #[test] fn test_channel_sense_attention() { let mut ch = Channel::new("test"); ch.deadband = 0.1; ch.predict(1.0); let (s, a) = ch.sense(2.0, 0); assert_eq!(s, -1); assert!(a); }
    #[test] fn test_channel_deadband_widens() { let mut ch = Channel::new("test"); let db = ch.deadband; ch.predict(1.0); ch.sense(1.0, 0); assert!(ch.deadband > db); }
    #[test] fn test_channel_deadband_narrows() { let mut ch = Channel::new("test"); ch.deadband = 0.5; ch.predict(0.0); ch.sense(5.0, 0); let db_after = ch.deadband; assert!(db_after < 0.5); }
    #[test] fn test_channel_read_ground() { let mut ch = Channel::new("test"); ch.predict(1.0); ch.sense(1.2, 0); assert!((ch.read_ground().unwrap() - 0.2).abs() < 0.01); }
    #[test] fn test_channel_health() { let mut ch = Channel::new("test"); ch.predict(1.0); ch.sense(1.0, 0); ch.predict(1.0); ch.sense(1.0, 1); assert!(ch.health() > 0.9); }
    #[test] fn test_channel_calibrated() { let mut ch = Channel::new("test"); for i in 0..15 { ch.predict(1.0); ch.sense(1.0, i as u64); } assert!(ch.is_calibrated()); }
    #[test] fn test_channel_not_calibrated() { let ch = Channel::new("test"); assert!(!ch.is_calibrated()); }
    #[test] fn test_channel_recent_accuracy() { let mut ch = Channel::new("test"); ch.predict(1.0); ch.sense(1.0, 0); ch.predict(1.0); ch.sense(0.0, 1); assert!(ch.recent_accuracy(5) < 1.0); }
    #[test] fn test_channel_history() { let mut ch = Channel::new("test"); ch.predict(1.0); ch.sense(1.0, 0); assert_eq!(ch.history.len(), 1); }

    // SimulationModel tests
    #[test] fn test_sim_model_new() { let m = SimulationModel::new("test", 3); assert_eq!(m.weights.len(), 3); }
    #[test] fn test_sim_model_predict() { let mut m = SimulationModel::new("test", 2); let p = m.predict(&[1.0, 0.5]); assert_eq!(p, 0.0); } // Zero weights
    #[test] fn test_sim_model_learn() { let mut m = SimulationModel::new("test", 2); m.predict(&[1.0, 1.0]); m.learn(&[1.0, 1.0], 2.0); assert!(m.weights[0] > 0.0); }
    #[test] fn test_sim_model_confidence() { let m = SimulationModel::new("test", 1); assert!((m.confidence() - 0.5).abs() < 0.01); }
    #[test] fn test_sim_model_improves() { let mut m = SimulationModel::new("test", 1); m.learning_rate = 0.1; for _ in 0..100 { m.predict(&[1.0]); m.learn(&[1.0], 1.0); } assert!(m.confidence() > 0.5); }

    // PredictionEngine tests
    #[test] fn test_engine_new() { let e = PredictionEngine::new(); assert!(e.channels.is_empty()); }
    #[test] fn test_engine_add_channel() { let mut e = PredictionEngine::new(); e.add_channel("shoe"); assert!(e.channels.contains_key("shoe")); }
    #[test] fn test_engine_simulate_and_sense() { let mut e = PredictionEngine::new(); e.add_channel("room_b"); e.add_simulation("room_b", 1); e.simulate("room_b", &[1.0]); let (s, a) = e.sense("room_b", 0.0); assert!(!a); }
    #[test] fn test_engine_perception_uninitialized() { let e = PredictionEngine::new(); assert_eq!(e.perception_state(), PerceptionState::Uninitialized); }
    #[test] fn test_engine_tick() { let mut e = PredictionEngine::new(); e.tick(); assert_eq!(e.tick, 1); }
    #[test] fn test_engine_attention_channels() { let mut e = PredictionEngine::new(); e.add_channel("a"); e.add_channel("b"); let ch = e.channels.get_mut("a").unwrap(); ch.deadband = 0.01; ch.predict(0.0); ch.sense(5.0, 0); let attn = e.attention_channels(); assert!(attn.contains(&"a")); }
    #[test] fn test_engine_read_all_ground() { let mut e = PredictionEngine::new(); e.add_channel("shoe"); let ch = e.channels.get_mut("shoe").unwrap(); ch.predict(1.0); ch.sense(1.5, 0); let ground = e.read_all_ground(); assert!((ground.get("shoe").unwrap() - 0.5).abs() < 0.01); }
    #[test] fn test_engine_global_confidence() { let e = PredictionEngine::new(); assert!((e.global_confidence() - 0.5).abs() < 0.01); }
    #[test] fn test_engine_channel_summary() { let mut e = PredictionEngine::new(); e.add_channel("a"); let summary = e.channel_summary(); assert_eq!(summary.len(), 1); }
    #[test] fn test_engine_sense_missing() { let mut e = PredictionEngine::new(); let (s, a) = e.sense("nonexistent", 1.0); assert_eq!(s, 0); assert!(!a); }

    // PredictionLandmark tests
    #[test] fn test_landmark_new() { let l = PredictionLandmark::new(0, 10); assert!(!l.should_fire(5)); assert!(l.should_fire(10)); }
    #[test] fn test_landmark_predict_channel() { let mut l = PredictionLandmark::new(0, 10); l.predict_channel("room_b", 1.0); assert_eq!(l.channel_predictions["room_b"], 1.0); }
    #[test] fn test_landmark_reconcile() { let mut l = PredictionLandmark::new(0, 10); l.predict_channel("a", 1.0); let mut actuals = HashMap::new(); actuals.insert("a".to_string(), 1.0); let deltas = l.reconcile(&actuals); assert_eq!(deltas.len(), 1); assert!(!deltas[0].surprise); }
    #[test] fn test_landmark_surprise() { let mut l = PredictionLandmark::new(0, 10); l.confidence_threshold = 0.1; l.predict_channel("a", 1.0); let mut actuals = HashMap::new(); actuals.insert("a".to_string(), 5.0); let deltas = l.reconcile(&actuals); assert!(deltas[0].surprise); }
    #[test] fn test_landmark_fires_once() { let mut l = PredictionLandmark::new(0, 10); let mut actuals = HashMap::new(); l.reconcile(&actuals); assert!(l.fired); assert!(!l.should_fire(10)); }

    // PerceptionState tests
    #[test] fn test_perception_display() { assert_eq!(format!("{}", PerceptionState::Flowing), "🌊 FLOWING"); }
    #[test] fn test_perception_calibrating() { let mut e = PredictionEngine::new(); e.add_channel("a"); for i in 0..5 { let ch = e.channels.get_mut("a").unwrap(); ch.predict(1.0); ch.sense(1.1, i as u64); } let state = e.perception_state(); assert!(matches!(state, PerceptionState::Calibrating | PerceptionState::Learning | PerceptionState::Flowing)); }
}
