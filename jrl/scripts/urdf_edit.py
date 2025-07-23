import re

with open("jrl/urdfs/iiwa7_L/lbr1.urdf", "r") as file:
    content = file.read()

# Ersetze visual
content = re.sub(r'mesh filename="iiwa7/visual/', r'mesh filename="urdfs/iiwa7_L/meshes/visual/', content)

# Ersetze collision
content = re.sub(r'mesh filename="iiwa7/collision/', r'mesh filename="urdfs/iiwa7_L/meshes/collision/', content)

with open("jrl/urdfs/iiwa7_L/iiwa7_L_updated.urdf", "w") as file:
    file.write(content)
