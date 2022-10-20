import sys
sys.path.append('.')
import argparse
from datetime import datetime
import wandb
import numpy as np
import torch
from TrainingNavigator.DemoImitation.ImitationUtils import demos_to_tensor, get_policy_to_train, normalize_dataset, \
    prepare_data_loader, test_loss


def save_actor(actor, val_loss, step, maze: str):
    date_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    print("Saving model at step: ", step, " with val loss: ", val_loss, " at time: ", date_time)
    save_path = "./TrainingNavigator/DemoImitation/actors/" + args.maze + "_" + date_time + ".pt"
    torch.save(actor, save_path)

    artifact = wandb.Artifact(date_time, type="actor" + maze)
    artifact.add_file(save_path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # env parameters:
    parser.add_argument("--maze", type=str, required=True)
    parser.add_argument("--action_radius_high", type=float, default=2.5)
    parser.add_argument("--action_radius_low", type=float, default=0.3)
    parser.add_argument("--maze_size", type=int, default=10)

    # hyper parameters:
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_train_trajectories", type=int, default=0
                        , help="number of train trajectories to use. 0 means all, if there are more than"
                               "this number, all is used as well")

    # training parameters:
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--validation_freq", type=int, default=200)

    args = parser.parse_args()
    demo_traj_path = "./TrainingNavigator/workspaces/" + args.maze + "/trajectories_train.npz"
    val_traj_path = "./TrainingNavigator/workspaces/" + args.maze + "/trajectories_validation.npz"
    maze_size = (args.maze_size, args.maze_size)

    wandb.init(project="NavigatorImitation", entity="robomaze", config=vars(args), name=args.run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = get_policy_to_train(maze_size, args.learning_rate).to(device)
    actor = policy.actor

    data_loader_train = prepare_data_loader(demo_traj_path, num_trajectories=args.num_train_trajectories,
                                            batch_size=args.batch_size, device=device,
                                            action_radius_high=args.action_radius_high,
                                            action_radius_low=args.action_radius_low, maze_size=maze_size,
                                            shuffle=True, )
    data_loader_val = prepare_data_loader(val_traj_path, batch_size=0, device=device,
                                          action_radius_high=args.action_radius_high,
                                          action_radius_low=args.action_radius_low, maze_size=maze_size,
                                          shuffle=False, )

    best_val_loss = np.inf
    for step in range(args.num_steps):
        obs, actions = next(iter(data_loader_train))
        actor_out = actor(obs)
        loss = torch.mean((actor_out - actions) ** 2)
        actor.optimizer.zero_grad()
        loss.backward()
        actor.optimizer.step()

        wandb.log({"train_loss": loss.item()}, step=step)

        if step % args.validation_freq == 0:
            val_loss, r_mean_err, next_direction_mean_err, next_heading_mean_err = test_loss(actor, data_loader_val)
            wandb.log({"val_loss": val_loss.item()}, step=step)

            # unscale r:
            r_mean_err = \
                (r_mean_err + 1) * (args.action_radius_high - args.action_radius_low) / 2 + args.action_radius_low
            # convert angles to degrees:
            next_direction_mean_err *= 180
            next_heading_mean_err *= 180
            wandb.log({"r_mean_err": r_mean_err.item()}, step=step)
            wandb.log({"next_direction_mean_err": next_direction_mean_err.item()}, step=step)
            wandb.log({"next_heading_mean_err": next_heading_mean_err.item()}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary["best_val_loss"] = best_val_loss
                save_actor(actor, val_loss.item(), step, args.maze)


    wandb.finish()