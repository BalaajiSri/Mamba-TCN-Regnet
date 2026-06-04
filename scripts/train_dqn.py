import os
import pathlib
import sys
from argparse import ArgumentParser

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from data_utils.dataset import DataConverter
from rl.dqn import DQNAgent, ReplayBuffer, Transition
from rl.environment import TradingEnvironment
from utils import io_tools


ROOT = io_tools.get_root(__file__, num_returns=2)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="alphaseek_dqn_1d", type=str)
    parser.add_argument("--signal_config", default="alphaseek_hybrid_1d", type=str)
    parser.add_argument("--signal_ckpt", default=None, type=str)
    parser.add_argument("--split", default="train", choices={"train", "val", "test"})
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser.parse_args()


def load_signal_module(signal_config_name, checkpoint_path, device):
    if checkpoint_path is None:
        return None
    training_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/training/{signal_config_name}.yaml")
    arch_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/models/archs.yaml")
    model_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/models/{arch_config[training_config['model']]}")
    model_config["params"]["num_features"] = len(
        feature_columns(
            use_volume=training_config.get("use_volume", False),
            additional_features=training_config.get("additional_features", []),
        )
    )
    model_class = io_tools.get_obj_from_str(model_config["target"])
    model = model_class.load_from_checkpoint(checkpoint_path, map_location=device, **model_config["params"])
    model.to(device)
    model.eval()
    return model


def feature_columns(use_volume: bool, additional_features: list[str]):
    columns = ["Timestamp", "Open", "High", "Low", "Close"]
    if use_volume:
        columns.append("Volume")
    columns.extend(additional_features)
    return columns


def encode_state(model, state, device):
    if model is None:
        return state.reshape(-1)
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        encoded = model.encode(state_tensor)
    return encoded.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main():
    args = parse_args()
    device = torch.device(args.device)

    rl_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/rl/{args.config}.yaml")
    data_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/data_configs/{rl_config['data_config']}.yaml")

    converter = DataConverter(data_config)
    train_df, val_df, test_df = converter.get_data()
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    data = split_map[args.split]

    signal_training_config = io_tools.load_config_from_yaml(f"{ROOT}/configs/training/{args.signal_config}.yaml")
    columns = feature_columns(
        use_volume=signal_training_config.get("use_volume", False),
        additional_features=signal_training_config.get("additional_features", []),
    )

    env = TradingEnvironment(
        data=data,
        feature_columns=columns,
        window_size=rl_config["window_size"],
        transaction_cost=rl_config["transaction_cost"],
        initial_cash=rl_config["initial_cash"],
        allow_short=rl_config["allow_short"],
        max_steps=rl_config["max_steps_per_episode"],
    )

    signal_model = load_signal_module(args.signal_config, args.signal_ckpt, device) if args.signal_ckpt else None
    initial_state = env.reset()
    encoded_state = encode_state(signal_model, initial_state, device)

    agent = DQNAgent(
        state_dim=int(encoded_state.shape[0]),
        hidden_dim=rl_config["hidden_dim"],
        lr=rl_config["lr"],
        gamma=rl_config["gamma"],
        tau=rl_config["tau"],
        double_dqn=rl_config["double_dqn"],
        dueling=rl_config["dueling"],
        device=str(device),
    )
    buffer = ReplayBuffer(rl_config["buffer_size"])

    output_dir = pathlib.Path(args.output_dir or f"{ROOT}/runs/rl/{args.config}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon = rl_config["epsilon_start"]
    all_rewards = []
    all_losses = []

    for episode in range(rl_config["episodes"]):
        state = env.reset()
        encoded = encode_state(signal_model, state, device)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.act(encoded, epsilon)
            step = env.step(action)
            next_encoded = encode_state(signal_model, step.observation, device) if not step.done else np.zeros_like(encoded)
            buffer.push(
                Transition(
                    state=encoded,
                    action=action,
                    reward=step.reward,
                    next_state=next_encoded,
                    done=float(step.done),
                )
            )
            encoded = next_encoded
            episode_reward += step.reward
            done = step.done

            if len(buffer) >= max(rl_config["batch_size"], rl_config["warmup_steps"]):
                loss = agent.update(buffer.sample(rl_config["batch_size"]))
                all_losses.append(loss)

        epsilon = max(rl_config["epsilon_end"], epsilon * rl_config["epsilon_decay"])
        all_rewards.append(episode_reward)
        print(f"episode={episode + 1} reward={episode_reward:.4f} epsilon={epsilon:.4f}")

    torch.save(agent.policy_net.state_dict(), output_dir / "dqn_policy.pt")
    np.save(output_dir / "episode_rewards.npy", np.asarray(all_rewards, dtype=np.float32))
    np.save(output_dir / "losses.npy", np.asarray(all_losses, dtype=np.float32))
    print(f"Saved RL artifacts to {output_dir}")


if __name__ == "__main__":
    main()
