import trimesh

# Parameter: Zylinder-Höhe und Radius
height = 0.12     # Länge des Zylinders in URDF
radius = 0.055    # Radius des Zylinders in URDF

# Erzeuge Zylinder (Z-Achse als Längsachse)
cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=64)

# Optional: nach oben verschieben, damit Ursprung an der Basis liegt (nicht Mittelpunkt)
cylinder.apply_translation([0, 0, height / 2])

# Speichern als STL
cylinder.export('lbr1_gripper_cylinder.stl')
