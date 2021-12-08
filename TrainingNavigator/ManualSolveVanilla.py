import math
from Utils import get_vanilla_navigator_env
import argparse


parser = argparse.ArgumentParser(description="Solve the vanilla maze with fixed manually generated actions")
parser.add_argument('--to_vid', dest='to_vid', action='store_const', const=True, default=False)

args = parser.parse_args()

nav_env = get_vanilla_navigator_env(show_gui=not args.to_vid)
nav_env.visualize_mode(not args.to_vid)

actions = [(3, -math.pi/6)] * 5

obs = nav_env.reset()
nav_env.maze_env.reset(create_video=args.to_vid, video_path="manualVanilla.avi")

for i in range(len(actions)):
    obs, reward, _, _ = nav_env.step(actions[i])
    print("reward:", reward)

