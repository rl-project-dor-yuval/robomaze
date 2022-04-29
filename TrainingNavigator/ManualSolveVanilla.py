import math
from Utils import get_vanilla_navigator_env
import argparse
import time

parser = argparse.ArgumentParser(description="Solve the vanilla maze with fixed manually generated actions")
parser.add_argument('--to_vid', dest='to_vid', action='store_const', const=True, default=False)

args = parser.parse_args()
stepper_path = "StepperAgent.pt"
nav_env = get_vanilla_navigator_env(show_gui=not args.to_vid, stepper_path=stepper_path)
nav_env.visualize_mode(not args.to_vid)

# actions = [(2, -math.pi / 6)] * 1


for i in range(5):
    obs = nav_env.reset()
    nav_env.maze_env.reset(create_video=args.to_vid, video_path="manualVanilla.avi")
    is_done = False

    while not is_done:
        user_action = input("Enter radius, theta as R,Theta in degrees")
        nxt_act = [float(x) for x in user_action.split(",")]
        nxt_act[1] = math.radians(nxt_act[1])

        obs, reward, is_done, _ = nav_env.step(nxt_act)
        print("reward:", reward)
