import pybullet as p


class Ant:
    def __init__(self, position3d):
        self.start_position = position3d

        # load ant and save it's initial orientation,
        # for now this will be the initial orientation always
        self.uid = p.loadMJCF("data/ant.xml")[0]
        self.initial_orientation = p.getBasePositionAndOrientation(self.uid)[1]

        self.reset()

        # TODO set colors

    def reset(self):
        p.resetBasePositionAndOrientation(self.uid,
                                          self.start_position,
                                          self.initial_orientation)

    def action(self, action):
        # TODO implement
        pass
