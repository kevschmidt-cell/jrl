from klampt import WorldModel, vis

# Lade URDF in Klampt
world = WorldModel()
robot = world.loadRobot("urdfs/dual_iiwa7/dual_iiwa7.urdf")


# Visualisieren
vis.add("world", world)
vis.show()
input("Drücke Enter zum Beenden...")
vis.kill()
