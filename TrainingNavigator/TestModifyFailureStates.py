from pathlib import Path
import os, time, sys

sys.path.append('.')
from TrainingNavigator.TD3MP import TD3MP
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from Utils import blackwhiteswitch
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.EnvAttributes import Rewards

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_scaled_action(obs, agent, env, device):
    obs_ = torch.from_numpy(obs).unsqueeze(0).to(device)

    with torch.no_grad():
        action = agent(obs_)
        action = action.squeeze(0).to('cpu').numpy()
        # scale action
        action = action.clip(-1, 1)
        low0, high0 = env.action_space.low[0], env.action_space.high[0]
        action[0] = low0 + (0.5 * (action[0] + 1.0) * (high0 - low0))
        action[1] = action[1] * env.action_space.high[1]

    return action


if __name__ == '__main__':
    maze_map = blackwhiteswitch('TrainingNavigator/workspaces/bottleneck/bottleneck.png')
    start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneck/test_workspaces.npy') / 10
    agent_path = 'TrainingNavigator/logs/bufferSize33k_demoprob02/saved_model/model_250000.zip'
    ws_id = 29

    # prepare environment:
    raise NotImplementedError("Fix to new workspaces")
    maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map,
                       show_gui=True)

    nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=start_goal_pairs,
                                         maze_env=maze_env,
                                         epsilon_to_hit_subgoal=0.25,
                                         max_vel_in_subgoal=999,
                                         rewards=Rewards(target_arrival=1, collision=-0.02, fall=-1, idle=-0.001, ),
                                         done_on_collision=False,
                                         max_stepper_steps=100,
                                         max_steps=30,
                                         velocity_in_obs=False,
                                         stepper_agent='TrainingNavigator/StepperAgents/TorqueStepperF1500.pt',
                                         stepper_radius_range=(0.3, 2),
                                         wall_hit_limit=10, )
    nav_env.visualize_mode(True, fps=120)

    # load agent:
    agent = TD3MP.load(agent_path, env=nav_env).policy.actor.eval().to(device)

    results_dir = 'TrainingNavigator/NavigatorTests/TestModifyFailureStates'

    # reproduce failed workspace:
    obs = prev_obs = nav_env.reset(start_goal_pair_idx=ws_id, create_video=False, video_path=results_dir + '/original.gif')
    done = False
    steps = 0
    while done is False:
        action = get_scaled_action(obs, agent, nav_env, device)
        prev_obs = obs
        obs, reward, done, info = nav_env.step(action)
        steps += 1

    failure_step = steps - 1
    original_failure_obs = prev_obs
    original_last_action = action
    print(f"played original workspace, failed at step:{failure_step}, last action:{original_last_action}, "
          f"observation at failure:{original_failure_obs}")
    print("")

    # play again change state at failed step to canonical joint position:
    obs = nav_env.reset(start_goal_pair_idx=ws_id, create_video=False,
                        video_path=results_dir + '/changed_to_canonical_joints.gif')
    done = False
    steps = 0
    for i in range(failure_step):
        action = get_scaled_action(obs, agent)

        obs, reward, done, info = nav_env.step(action)
        steps += 1

    predicted_action = get_scaled_action(obs, agent)

    # make sure we are in the same state as before:
    print(f"played original workspace, stopping at step:{steps}, predicted action:{predicted_action}, "
          f"observation:{obs}")
    assert np.all(obs == original_failure_obs)
    assert np.all(predicted_action == original_last_action)

    time.sleep(2)
    # (very) dirty change the state:
    from MazeEnv.Ant import JOINTS_INDICES, INIT_JOINT_STATES

    uid = maze_env._ant.uid

    # for joint, state in zip(JOINTS_INDICES, INIT_JOINT_STATES):
    #     maze_env._pclient.resetJointState(uid, joint, state)

    # maze_env._pclient.resetBaseVelocity(uid, [0, 0, 0], [0, 0, 0])

    pos, orientation = maze_env._pclient.getBasePositionAndOrientation(uid)
    new_orientation = maze_env._pclient.getEulerFromQuaternion(orientation)
    new_orientation = maze_env._pclient.getQuaternionFromEuler([new_orientation[0], new_orientation[1], -2.5])
    maze_env._pclient.resetBasePositionAndOrientation(uid, pos, new_orientation)
    time.sleep(2)

    # play the same action:
    obs, reward, done, info = nav_env.step(predicted_action)
    # keep playing:
    while not done:
        action = get_scaled_action(obs, agent)
        obs, reward, done, info = nav_env.step(action)

    print(info)
    time.sleep(5)
