import dataclasses
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy

from .env_runner import EnvRunner
from .utils import TASK_NAME_LIST
from .utils import TASK_WITH_VIDEO_DEMO
from .utils import EpisodeState
from .utils import RolloutRecorder
from .utils import check_args
from .utils import pack_buffer


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8011

    obs_horizon: int = 16
    max_steps: int = 1300
    save_dir: str = "runs/evaluation"
    overwrite: bool = False

    use_history: bool = True
    policy_name: str = "dummy_test"
    model_seed: int = 42
    model_ckpt_id: int = 80000

    # task control
    re_eval_tasks: str = ""  # tasks split by comma
    only_tasks: str = ""  # tasks split by comma
    exclude_tasks: str = ""  # tasks split by comma


class EpisodeEvaluator:
    def __init__(self, args: Args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def eval_each_episode(
        self,
        env_runner: EnvRunner,
        video_save_dir: Path,
    ) -> str:
        client = _websocket_client_policy.MMEVLAWebsocketClientPolicy(self.args.host, self.args.port)
        resp = client.reset()
        while not resp.get("reset_finished", False):
            time.sleep(0.1)

        epstate = EpisodeState()
        task_goal, recorder = self.init_episode(env_runner, epstate, video_save_dir)

        img, wrist_img, robot_state = epstate.get_current_obs()
        prompt = task_goal
        success_flag = "unknown"

        while True:
            if not epstate.action_plan:
                action_chunk = self.get_action_chunk(
                    client, epstate, img, wrist_img, robot_state, prompt, exec_horizon=self.args.obs_horizon
                )
                epstate.action_plan.extend(action_chunk)
                epstate.clear_buffers()

            action = epstate.action_plan.popleft()
            obs, stop_flag, success_flag = env_runner.step(action)
            epstate.count += 1

            if epstate.count > self.args.max_steps:
                success_flag = "timeout"
                break

            img, wrist_img, robot_state = obs

            epstate.add_observation(img, wrist_img, robot_state)
            recorder.record(
                image=img.copy(),
                wrist_image=wrist_img.copy(),
                state=robot_state.copy(),
                action=action.copy(),
            )

            if stop_flag:
                break

        if success_flag == "unknown":
            return "unknown"

        video_filename = (
            f"{env_runner.env_id}_ep{env_runner.episode_id}_{success_flag}_{task_goal}_{env_runner.difficulty}.mp4"
        )
        recorder.save_video(video_filename)

        return success_flag

    def init_episode(
        self,
        env_runner: EnvRunner,
        epstate: EpisodeState,
        video_save_dir: Path,
    ) -> tuple[str, RolloutRecorder]:
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]

        recorder = RolloutRecorder(video_save_dir, task_goal, fps=30)

        print(f"task_goal: {task_goal}")

        epstate.image_buffer.extend(pre_traj["images"])
        epstate.wrist_image_buffer.extend(pre_traj["wrist_images"])
        epstate.state_buffer.extend(pre_traj["states"])

        for i in range(len(pre_traj["images"])):
            recorder.record(
                image=pre_traj["images"][i].copy(),
                wrist_image=pre_traj["wrist_images"][i].copy(),
                state=pre_traj["states"][i].copy(),
                is_video_demo=env_runner.env_id in TASK_WITH_VIDEO_DEMO and i < len(pre_traj["images"]) - 1,
                subgoal=None,
            )

        epstate.exec_start_idx = len(epstate.image_buffer) - 1
        print(f"exec_start_idx: {epstate.exec_start_idx}")
        return task_goal, recorder

    def get_action_chunk(
        self,
        client,
        state: EpisodeState,
        img: np.ndarray,
        wrist_img: np.ndarray,
        robot_state: np.ndarray,
        prompt: str,
        exec_horizon: int,
    ) -> list:
        if self.args.use_history:
            resp = client.add_buffer(
                pack_buffer(
                    state.image_buffer,
                    state.state_buffer,
                    state.exec_start_idx,
                )
            )
            while not resp.get("add_buffer_finished", False):
                time.sleep(0.1)

        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": robot_state,
            "prompt": prompt,
        }

        action_chunk = client.infer(element)["actions"]
        return action_chunk[:exec_horizon]


def setup_save_directory(args: Args) -> Path:
    """Set up and validate save directories."""
    save_dir = Path(args.save_dir) / args.policy_name / f"ckpt{args.model_ckpt_id}" / f"seed{args.model_seed}"

    if save_dir.exists():
        if args.overwrite:
            shutil.rmtree(save_dir)
            print(f"we will overwrite the evaluation at {save_dir}")
        else:
            print("we will resume the evaluation")

    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def setup_log_dict(save_dir: Path, args: Args) -> dict:
    if os.path.exists(save_dir / "progress.json"):
        with open(save_dir / "progress.json") as f:
            log_dict = json.load(f)

    elif os.path.exists(save_dir / "log.json"):
        with open(save_dir / "log.json") as f:
            log_dict = json.load(f)
        log_dict.pop("success_rate", None)
        log_dict.pop("total_success_rate", None)
    else:
        log_dict = {}

    for task_name in log_dict:
        error_list = []
        for k, v in log_dict[task_name].items():
            if v == "error":
                error_list.append(k)
        for k in error_list:
            log_dict[task_name].pop(k)

    if args.re_eval_tasks:
        for task_name in args.re_eval_tasks.split(","):
            if task_name in log_dict:
                del log_dict[task_name]
                os.system(f"rm -f {save_dir / 'videos' / f'{task_name}_ep*.mp4'}")

    with open(save_dir / "progress.json", "w") as f:
        json.dump(log_dict, f, indent=2)

    return log_dict


def evaluate(args: Args):
    """Main evaluation function."""
    check_args(args)

    save_dir = setup_save_directory(args)
    video_save_dir = save_dir / "videos"

    log_dict = setup_log_dict(save_dir, args)

    if args.only_tasks:
        task_names = args.only_tasks.split(",")
    else:
        task_names = TASK_NAME_LIST

    if args.exclude_tasks:
        task_names = [task_name for task_name in task_names if task_name not in args.exclude_tasks.split(",")]
        for task in args.exclude_tasks.split(","):
            log_dict[task] = {str(i): False for i in range(50)}

    evaluator = EpisodeEvaluator(args, save_dir)

    while not os.path.exists(save_dir / "log.json"):
        for task_name in task_names:
            if task_name not in log_dict:
                log_dict[task_name] = {}

            env_runner = EnvRunner(task_name, video_save_dir, max_steps=args.max_steps)
            num_episodes = env_runner.num_episodes

            success_flag = "unknown"

            for episode_id in range(num_episodes):
                if str(episode_id) in log_dict[task_name]:
                    print(f"[robomme] episode {episode_id} already evaluated, skipping...")
                    continue

                env_runner.make_env(episode_id)
                print(f"\n[robomme] env for task {task_name} episode {episode_id} setup finished")

                try:
                    success_flag = evaluator.eval_each_episode(env_runner, video_save_dir)
                    if success_flag == "unknown":
                        log_dict[task_name][episode_id] = "error"
                    else:
                        log_dict[task_name][episode_id] = success_flag == "success"
                except Exception as e:
                    print(f"Error evaluating episode {episode_id} for task {task_name}: {e}")
                    log_dict[task_name][episode_id] = "error"

                env_runner.close_env()
                with open(save_dir / "progress.json", "w") as f:
                    json.dump(log_dict, f, indent=2)

                if success_flag == "unknown":
                    print("API calling error, aborting...")
                    return

            del env_runner
            time.sleep(1)

        try:
            final_results = {}
            final_results["success_rate"] = {
                task_name: sum(log_dict[task_name].values()) / len(log_dict[task_name].values())
                for task_name in log_dict.keys()
            }
            final_results["total_success_rate"] = sum(final_results["success_rate"].values()) / len(
                final_results["success_rate"].values()
            )
            with open(save_dir / "log.json", "w") as f:
                json.dump(final_results, f, indent=2)
        except Exception as e:
            print(f"Error saving final results: {e}")
            time.sleep(1)


def main() -> None:
    import tyro

    tyro.cli(evaluate)


if __name__ == "__main__":
    main()
