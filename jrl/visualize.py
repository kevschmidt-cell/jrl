from klampt import WorldModel, vis
import os

# Falls nötig:
os.chdir("/home/kevin/dev/jrl/jrl")  # Damit Meshpfade korrekt aufgelöst werden

world = WorldModel()
world.readFile("urdfs/iiwa7_L/iiwa7_L_updated.urdf")
robot = world.robot(0)

print(f"Geladene Links: {robot.numLinks()}")
for i in range(robot.numLinks()):
    link = robot.link(i)
    print(f"{i}: {link.getName()}  parent: {link.getParent()}")


# Qt entfernen oder alternativ:
# pip install --upgrade klampt → dann funktioniert vis.setBackend("GLUT")

vis.add("world", world)
vis.show()
input("Drücke Enter zum Beenden...")
vis.kill()
