from klampt import WorldModel, vis
from klampt.model import config
from jrl.robots import Iiwa7_L

import numpy as np

robot = Iiwa7_L()
q, pose = robot.sample_joint_angles_and_poses(1)
q = q[0]

# Klampt-Welt laden
world = WorldModel()
world.loadElement(robot.urdf_filepath)
klampt_robot = world.robot(0)

# Pose setzen
klampt_robot.setConfig(q.tolist())

# Visualisierung starten
vis.add("world", world)
vis.show()
input("Drücke Enter zum Schließen...")
vis.kill()
