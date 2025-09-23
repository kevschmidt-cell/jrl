import numpy as np
from jrl.robots import DualIiwa7

def main():
    # Dual-Arm initialisieren
    robot = DualIiwa7(verbose=True)

    print("=== DualIiwa7 Test ===")
    print(f"DoFs gesamt: {robot.ndof}")

    # Dummy-Konfiguration anlegen (alle Gelenke = 0.0)
    q = np.zeros(robot.ndof)
    print("q:", q)

    # Config splitten
    q_left, q_right = robot.split_config(q)
    print("q_left:", q_left)
    print("q_right:", q_right)

    # Config mergen
    q_merged = robot.merge_config(q_left, q_right)
    print("q_merged == q ?", np.allclose(q, q_merged))

    # Forward Kinematics
    T_left = robot.fk_left(q)
    T_right = robot.fk_right(q)
    print("EE Pose Left:\n", T_left)
    print("EE Pose Right:\n", T_right)

    # Collision-Check
    coll = robot.check_collision(q)
    print("Collision?", coll)

if __name__ == "__main__":
    main()
