import sys
from pathlib import Path

# ---------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.g1_fixed_body_throw_env import G1FixedBodyThrowEnv
from baselines.baseline_controller import BASELINE_ACTION_SCALE, BaselineController


def evaluate(num_episodes=10, base_seed=42):

    env = G1FixedBodyThrowEnv(
        learned_release=True,
        action_scale=BASELINE_ACTION_SCALE,
    )

    controller = BaselineController(
        env.n_arm,
        nominal_joint_target_rad=env.nominal_ctrl[env.arm_actuator_ids],
        action_scale=env.action_scale,
    )

    success = 0

    rewards = []
    final_distances = []
    best_distances = []
    release_times = []

    for ep in range(num_episodes):

        obs, info = env.reset(seed=base_seed + ep)

        done = False
        episode_reward = 0.0

        while not done:

            t = env.step_count * env.control_dt

            action = controller.act(t)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            done = terminated or truncated

        rewards.append(episode_reward)
        final_distances.append(info["dist_to_target"])
        best_distances.append(info["best_dist"])

        if info["released"]:
            release_times.append(info["release_time"])

        if info["success"]:
            success += 1

        print("=" * 50)
        print(f"Episode {ep+1}")
        print("=" * 50)
        print(f"Reward          : {episode_reward:.2f}")
        print(f"Landing Error   : {info['landing_error']:.3f} m" if info['landing_error'] is not None else "Landing Error   : not landed")
        print(f"Best XY Error   : {info['best_dist']:.3f} m")
        print(f"Released        : {info['released']}")
        print(f"Release Time    : {info['release_time']}")

    print("\n")
    print("=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"Episodes                : {num_episodes}")
    print(f"Success Rate            : {100*success/num_episodes:.1f}%")
    print(f"Average Reward          : {sum(rewards)/len(rewards):.2f}")
    print(f"Average XY Error        : {sum(final_distances)/len(final_distances):.3f} m")
    print(f"Average Best XY Error   : {sum(best_distances)/len(best_distances):.3f} m")

    if release_times:
        print(f"Average Release Time    : {sum(release_times)/len(release_times):.3f} s")


if __name__ == "__main__":
    evaluate(10)
