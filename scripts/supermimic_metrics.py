"""Shared TensorBoard metrics; plotting needs no simulator."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCORE_TAG = "RawEpisode/raw_rew_tracking_lin_vel"
OTHER_TAGS = {
    "mean_episode_reward": "Train/mean_reward",
    "yaw_tracking_episode_score": "RawEpisode/raw_rew_tracking_ang_vel",
    "foot_phase_episode_score": "RawEpisode/raw_rew_feet_phase",
    "action_rate_episode_penalty": "RawEpisode/raw_rew_penalty_action_rate",
    "penalty_scale": "Env/penalty_scale",
    "mean_episode_steps": "Train/mean_episode_length",
    "simulator_transitions": "Train/num_samples",
    "simulator_transitions_per_second": "Perf/total_fps",
    "collection_seconds": "Perf/collection_time",
    "learning_seconds": "Perf/learning_time",
}
