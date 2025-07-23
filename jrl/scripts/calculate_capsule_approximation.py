import torch
import numpy as np
import stl
import pathlib
import meshcat
import argparse

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from jrl.robots import ALL_ROBOT_NAMES


def capsule_volume_batch(c1: torch.Tensor, c2: torch.Tensor, r: torch.Tensor):
    """Compute the volume of a capsule given its end points and radius.

    Args:
        c1 [batch x 3]: First end point.
        c2 [batch x 3]: Second end point.
        r [batch]: Radius of the capsule.

    Returns:
        [batch]: Volume of the capsule.
    """

    h = torch.norm(c2 - c1, dim=1)
    return np.pi * h * (r**2) + (4 / 3) * np.pi * (r**3)


def point_capsule_distance_batch(p: torch.Tensor, c1: torch.Tensor, c2: torch.Tensor, r: torch.Tensor):
    """Compute the distance between a point and a capsule given its end points and radius.

    Args:
        p [batch x 3]: Point.
        c1 [batch x 3]: First end point.
        c2 [batch x 3]: Second end point.
        r [batch]: Radius of the capsule.

    Returns:
        [batch]: Distance between the point and the capsule.
    """

    pc1 = p - c1
    c2c1 = c2 - c1
    h = torch.clamp((pc1 * c2c1).sum(dim=1) / (c2c1 * c2c1).sum(dim=1), 0, 1)
    return torch.norm(pc1 - h.unsqueeze(1) * c2c1, dim=1) - r


def plot_sphere(ax, center, radius):
    if isinstance(center, torch.Tensor):
        center = center.cpu().numpy()
    if isinstance(radius, torch.Tensor):
        radius = radius.cpu().item()
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color="r")


def random_choice_optimal_capsule(vertices: torch.Tensor):
    print("Entering random_choice_optimal_capsule()")
    device = vertices.device
    nsamples = 1000
    bbox_min = vertices.min(dim=0).values
    bbox_max = vertices.max(dim=0).values
    bbox_diag = torch.norm(bbox_max - bbox_min)
    print("BBox min:", bbox_min)
    print("BBox max:", bbox_max)
    print("BBox diag:", bbox_diag)
    scale = bbox_diag.item()

    # Beispiel: Skaliert basierend auf Bounding Box des STL
    p1s = bbox_min + (bbox_max - bbox_min) * torch.rand(nsamples, 3, device=device)
    p2s = bbox_min + (bbox_max - bbox_min) * torch.rand(nsamples, 3, device=device)
    rs = bbox_diag * 0.37 * torch.rand(nsamples, device=device) + bbox_diag * 0.05

    print(f"random_choice_optimal_capsule() | Sampled {nsamples} capsules")
    print("Shape for distance computation:", nsamples * vertices.shape[0])

    dists = point_capsule_distance_batch(
        vertices.unsqueeze(0).expand(nsamples, -1, -1).reshape(-1, 3),
        p1s.unsqueeze(1).expand(-1, vertices.shape[0], -1).reshape(-1, 3),
        p2s.unsqueeze(1).expand(-1, vertices.shape[0], -1).reshape(-1, 3),
        rs.unsqueeze(1).expand(-1, vertices.shape[0]).reshape(-1),
    ).reshape(nsamples, vertices.shape[0])
    maxdists, _ = torch.max(dists, dim=1)
    margin = 0.02 * bbox_diag
    mask = maxdists < margin
    print(f"random_choice_optimal_capsule() | Found valid: {mask.sum().item()} of {nsamples}")

    if torch.any(mask):
        _, i = torch.min(capsule_volume_batch(p1s[mask], p2s[mask], rs[mask]), dim=0)
        p1, p2, r = p1s[mask][i], p2s[mask][i], rs[mask][i]
    else:
        # Fallback: besten Kandidaten mit minimalem Überschuss nehmen
        volume = capsule_volume_batch(p1s, p2s, rs)  # (nsamples,)
        target_margin = 0.02 * scale  # oder z. B. 0.03 * scale
        score = (maxdists - target_margin) ** 2 + 0.5 * volume  # Gewichtung anpassbar
        _, i = torch.min(score, dim=0)
        p1, p2, r = p1s[i], p2s[i], rs[i]


    return p1, p2, r


def lm_penalty_optimal_capsule(vertices: torch.Tensor, nruns=1, vis=None):
    best_p1, best_p2, best_r = None, None, None
    best_cost = float("inf")
    margin = 1e-3

    for i in range(nruns):
        try:
            p1_0 = torch.randn(3, device=vertices.device)
            p1_0 = 0.2 * p1_0 / torch.norm(p1_0)
            p2_0 = -p1_0
            r_0 = torch.abs(torch.randn(1, device=vertices.device))
            x = torch.cat((p1_0, p2_0, r_0), dim=0)

            def fg(x, mu, vertices):
                p1, p2, r = x[None, 0:3], x[None, 3:6], x[6:7]
                dists = point_capsule_distance_batch(
                    vertices,
                    p1.expand(vertices.shape[0], -1),
                    p2.expand(vertices.shape[0], -1),
                    r.expand(vertices.shape[0]),
                )
                return torch.cat(
                    (
                        capsule_volume_batch(p1, p2, r),
                        torch.clamp(mu * dists, min=0),
                    )
                )

            Jfn = torch.func.jacfwd(fg, argnums=0)

            xtol = 1e-6
            mu = 0.1
            outer_step = 0
            satisfied = False

            while not satisfied:
                inner_step = 0
                converged = False
                while not converged:
                    J = Jfn(x, mu, vertices)
                    A = J.t() @ J + (1e-6) * torch.eye(J.shape[1], device=J.device)
                    b = -J.t() @ fg(x, mu, vertices)
                    dx = torch.linalg.solve(A, b)
                    x = x + 1e-2 / mu * dx

                    if torch.norm(dx) < xtol:
                        converged = True
                        print(f"Converged in {inner_step} steps")

                    if inner_step > 5000:
                        raise RuntimeError(f"Did not converge in {inner_step} inner LM steps")

                    inner_step += 1

                    mins = torch.min(vertices, dim=0)[0]
                    maxs = torch.max(vertices, dim=0)[0]
                    x = torch.clamp(
                        x,
                        min=torch.cat([mins, mins, torch.tensor([0.01], device=x.device)]),
                        max=torch.cat([maxs, maxs, torch.tensor([1.0], device=x.device)]),
                    )

                if torch.all(fg(x, 1, vertices)[1:] <= margin):
                    satisfied = True
                    print(f"Satisfied in {outer_step} outer steps")

                if outer_step > 100:
                    break

                mu *= 2
                outer_step += 1

            p1, p2, r = x[0:3], x[3:6], x[6]
            cost = capsule_volume_batch(p1.reshape(-1, 3), p2.reshape(-1, 3), r.reshape(-1))
            if cost < best_cost:
                best_p1, best_p2, best_r = p1.detach(), p2.detach(), r.detach()
                best_cost = cost
                print("New best capsule (satisfies constraint)" if satisfied else "New best capsule (but does NOT satisfy constraint)")

        except Exception as e:
            print(f"Run {i+1}/{nruns} failed: {e}")
            continue

    if best_r is None:
        raise RuntimeError("Capsule optimization failed: no valid solution found (even approximate)")

    return best_p1, best_p2, best_r + margin





def stl_to_capsule(stl_path: str, outdir: pathlib.PosixPath, vis=None):
    linkname = stl_path.stem
    txt_path = outdir / f"{linkname}.txt"
    if txt_path.exists():
        print(f"\nStl '{stl_path}' has been converted already - skipping")
        return

    print(f"\nstl_to_capsule() | Approximating {stl_path}")
    stl_mesh_geom = meshcat.geometry.StlMeshGeometry.from_file(stl_path)
    vis["mesh"].set_object(stl_mesh_geom)
    mesh = stl.mesh.Mesh.from_file(stl_path)
    vertices = mesh.vectors.reshape(-1, 3)
    vertices = torch.tensor(vertices)

    try:
        p1, p2, r = lm_penalty_optimal_capsule(vertices, vis=vis)
    except RuntimeError as e:
        print(f"Optimization failed for {linkname}, falling back to random capsule: {e}")
        try:
            torch.cuda.empty_cache()
            vertices = vertices.cpu()
            p1, p2, r = random_choice_optimal_capsule(vertices)
            print("Fallback capsule result:")
            print("p1", p1)
            print("p2", p2)
            print("r", r)
        except Exception as e2:
            print(f"Fallback capsule computation failed: {e2}")
            return  # Kein Speichern, wenn Kapsel nicht funktioniert

    # Hier: Visualisierung & Speicherung
    try:
        p1_np = p1.cpu().numpy()
        p2_np = p2.cpu().numpy()
        r_val = float(r.cpu())

        # Visualisierung wie gehabt
        figure = plt.figure()
        axes = figure.add_subplot(projection="3d")
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
        plot_sphere(axes, p1_np, r_val)
        plot_sphere(axes, p2_np, r_val)
        axes.auto_scale_xyz(mesh.points.flatten(), mesh.points.flatten(), mesh.points.flatten())

        img_path = outdir / f"{linkname}.png"
        print(f"Rendering to {img_path}")
        plt.savefig(img_path)

        txt_path = outdir / f"{linkname}.txt"
        print(f"Saving capsule to {txt_path}")
        with open(txt_path, "w") as f:
            f.write(f"{p1_np[0]}, {p1_np[1]}, {p1_np[2]}, {p2_np[0]}, {p2_np[1]}, {p2_np[2]}, {r_val}\n")

        print(f"Done with '{stl_path}'")

    except Exception as final_err:
        print(f"Saving/rendering failed: {final_err}")



"""
uv run python scripts/calculate_capsule_approximation.py --visualize --robot_name=iiwa14
"""


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--visualize", action="store_true")
    argparser.add_argument("--robot_name", type=str, required=True)
    args = argparser.parse_args()

    assert args.robot_name in ALL_ROBOT_NAMES

    vis = None
    if args.visualize:
        vis = meshcat.Visualizer()
        vis.open()

    outdir = pathlib.Path(f"jrl/urdfs/{args.robot_name}/capsules")
    outdir.mkdir(exist_ok=True)
    for stl_path in pathlib.Path(f"jrl/urdfs/{args.robot_name}/meshes/onrobot_2fg7").glob("*.stl"):
        stl_to_capsule(stl_path, outdir, vis)