from klampt import WorldModel, vis
from klampt.model import config
from jrl.robots import Iiwa7_L, Simple_Robot_01
import os

import numpy as np
# Falls nötig:
os.chdir("/home/kevin/dev/jrl/jrl")  # Damit Meshpfade korrekt aufgelöst werden
robot = Simple_Robot_01()
q, pose = robot.sample_joint_angles_and_poses(1)
q = q[0]

# Klampt-Welt laden
world = WorldModel()
world.loadElement(robot.urdf_filepath)
klampt_robot = world.robot(0)

print("=== Debug Robot Config ===")
print("Num links:   ", klampt_robot.numLinks())
print("Num drivers: ", klampt_robot.numDrivers())
print("Expected config length: ", len(klampt_robot.getConfig()))
print("Your q length: ", len(q))
print("q: ", q)
print("==========================")

# Volle Config holen
q_full = klampt_robot.getConfig()

# Deine 3 Gelenke an die richtigen Indizes eintragen
q_full[1] = q[0]  # link_1 joint
q_full[2] = q[1]  # link_2 joint
q_full[3] = q[2]  # link_3 joint
# fixed joints (0 und 4) bleiben wie sie sind

# Pose setzen
klampt_robot.setConfig(q_full)

# Visualisierung starten
vis.add("world", world)
vis.show()
input("Drücke Enter zum Schließen...")
vis.kill()

