

class MazeSize:
    """
    3 different sizes that could be set for the maze
    """
    SQUARE5 = (5, 5)
    SQUARE10 = (10, 10)
    SQUARE15 = (15, 15)
    SQUARE20 = (20, 20)
    SMALL = (5, 10)
    MEDIUM = (10, 15)
    LARGE = SQUARE20


class Rewards:
    def __init__(self, target_arrival=1, collision=-1, timeout=0, idle=0):
        """
        The collection of rewards and their values
        :param target_arrival: the reward's value for arriving the target
        :param collision: the reward's value for a collision
        :param timeout: the reward's value for timeout
        :param idle: the reward for a time step where nothing else happens
        """
        self.idle = idle
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout


class ObservationsDefinition:
    observations_opts = {"joint_state", "robot_loc", "robot_target_loc"}

    def __init__(self, observations: list = ["joint_state", "robot_loc", "robot_target_loc"]):
        for ob in observations:
            if ob not in self.observations_opts:
                raise ValueError

        self.observations = observations

