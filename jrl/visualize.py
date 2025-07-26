from klampt import WorldModel, vis
import os

# Falls nötig:
os.chdir("/home/kevin/dev/jrl/jrl")  # Damit Meshpfade korrekt aufgelöst werden

world = WorldModel()
world.loadRobot("urdfs/iiwa7_R/iiwa7_R_updated.urdf")

robot = world.robot(0)
print("Roboter geladen:", robot.getName())
print("  Gelenke:", robot.numLinks())

# Qt entfernen oder alternativ:
# pip install --upgrade klampt → dann funktioniert vis.setBackend("GLUT")

vis.add("world", world)
vis.show()
input("Drücke Enter zum Beenden...")
vis.kill()
