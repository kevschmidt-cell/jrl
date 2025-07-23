import os
import pyassimp

input_dir = "jrl/urdfs/iiwa7_L/meshes/onrobot_2fg7"
output_dir = os.path.join(input_dir, "converted_stl")
os.makedirs(output_dir, exist_ok=True)

converted = 0
skipped = 0

for fname in os.listdir(input_dir):
    if fname.endswith(".dae"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname.replace(".dae", ".stl"))
        try:
            scene = pyassimp.load(in_path)
            pyassimp.export(scene, out_path, file_type="stl")
            pyassimp.release(scene)
            print(f"✅ Converted: {fname} → {out_path}")
            converted += 1
        except Exception as e:
            print(f"❌ Failed to convert {fname}: {e}")
            skipped += 1

print(f"\nDone. {converted} converted, {skipped} failed.")
