import numpy as np
import trimesh
from trimesh.creation import cylinder, icosphere

def load_capsule_txt(txt_path):
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        vals = list(map(float, line.split(',')))
        p1 = np.array(vals[0:3])
        p2 = np.array(vals[3:6])
        r = vals[6]
        return p1, p2, r

def create_capsule_mesh(p1, p2, radius, sphere_subdiv=3):
    height = np.linalg.norm(p2 - p1)
    direction = (p2 - p1) / height
    center = (p1 + p2) / 2.0

    # Create cylinder
    cyl = cylinder(radius=radius, height=height, sections=32)
    cyl.apply_translation([0, 0, height / 2])

    # Align cylinder to vector
    z_axis = np.array([0, 0, 1])
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    cyl.apply_transform(rotation_matrix)
    cyl.apply_translation(center - cyl.center_mass)

    # Create spheres at ends
    sph1 = icosphere(radius=radius, subdivisions=sphere_subdiv)
    sph2 = icosphere(radius=radius, subdivisions=sphere_subdiv)
    sph1.apply_translation(p1)
    sph2.apply_translation(p2)

    # Combine all
    capsule = trimesh.util.concatenate([cyl, sph1, sph2])
    return capsule

# === Hauptskript ===
if __name__ == "__main__":
    txt_path = "jrl/urdfs/iiwa7_L/capsules/2fg7_module.txt"        # Pfad zur Kapseldefinition
    stl_path = "jrl/urdfs/iiwa7_L/capsules/2fg7_module_capsule.stl"  # Zieldatei

    p1, p2, r = load_capsule_txt(txt_path)
    mesh = create_capsule_mesh(p1, p2, r)
    mesh.export(stl_path)
    print(f"Capsule STL saved to: {stl_path}")
