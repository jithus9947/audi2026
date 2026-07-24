"""Hand/palm geometry helpers for the G1 target-throw task.

Ground truth from inspecting assets/unitree_g1/g1.xml directly (not assumed):
this G1 model has NO separate palm body and NO finger joints. The "hand" is a
single rigid, jointless mesh (`right_rubber_hand`) fixed to
`right_wrist_yaw_link`, and that mesh is VISUAL ONLY -- contype=0/conaffinity=0,
no collision. The wrist body's actual collision geometry is a small mesh near
the wrist joint itself, nowhere near where the ball sits. So:

  - "Palm orientation" cannot be independently controlled or measured -- it IS
    right_wrist_yaw_link's orientation, full stop. There's no extra DOF.
  - "Palm forward" is that body's local +x axis. This is not a guess: the
    existing (untouched) ball-hold weld in scene_throw.xml already offsets the
    ball along local +x from this body, and the visual hand mesh itself is
    also offset along +x -- both independently confirm the convention.
  - "Palm normal" has no strong physical meaning for a fingerless mitten mesh
    (there's no distinct "back of hand" vs "palm side" in the geometry). It's
    defined here as local +z by convention, for diagnostics/visualization
    only -- flagged clearly rather than presented as if it were measured.
  - The ball previously overlapped the visual hand mesh (not a physics bug --
    that mesh never collides with anything) because its hold offset was a
    hand-set constant. compute_ball_hold_offset() replaces that guess with a
    value derived from the mesh's own vertex bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _mesh_local_bbox(model, geom_id: int):
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None
    adr = model.mesh_vertadr[mesh_id]
    num = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[adr : adr + num]
    return verts.min(axis=0), verts.max(axis=0)


def find_hand_mesh_geom(model, hand_body_id: int) -> int:
    """The geom on ``hand_body_id`` that extends furthest along local +x --
    empirically (and by the model's own naming/asset order) this is the
    visual hand mesh, not the small wrist collision mesh near the origin."""
    best_geom, best_x = -1, -np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != hand_body_id:
            continue
        bbox = _mesh_local_bbox(model, g)
        if bbox is None:
            continue
        local_max_x = model.geom_pos[g][0] + bbox[1][0]
        if local_max_x > best_x:
            best_x, best_geom = local_max_x, g
    return best_geom


def compute_ball_hold_offset(
    model,
    hand_body_id: int,
    ball_radius: float,
    contact_margin: float = 0.002,
    extra_offset_local: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """(pos_xyz, quat_wxyz) for the ball relative to ``hand_body_id`` such that
    it rests just past the hand mesh's fingertip end -- not embedded in it,
    not floating away. Derived from the mesh's actual vertex bounding box:

        offset_x = hand_mesh_max_x + ball_radius + contact_margin
        offset_y, offset_z = centered on the mesh's own cross-section there

    Verified by rendering (see envs/g1_target_throw_env.py's docstring/PR
    notes): this computed offset was checked against the actual mesh
    visually, not just trusted algebraically.

    ``extra_offset_local`` (forward, lateral, vertical) is added on top of
    the geometry-derived offset -- e.g. envs/right_arm_config.py's calibrated
    lateral shift, where local +y is the palm's own lateral axis and was
    independently verified (algebraically and by rendering, see the change
    report) to point toward the robot's own LEFT at a representative
    extended-throw pose. Zero by default, i.e. the pure geometry result.
    """
    geom_id = find_hand_mesh_geom(model, hand_body_id)
    extra = np.asarray(extra_offset_local, dtype=np.float64)
    if geom_id < 0:
        # No mesh geom found on this body (shouldn't happen for this model) --
        # fall back to the scene XML's original constant rather than guessing.
        return np.array([0.16, 0.0, 0.0]) + extra, np.array([1.0, 0.0, 0.0, 0.0])
    bbox_min, bbox_max = _mesh_local_bbox(model, geom_id)
    geom_pos = model.geom_pos[geom_id]
    offset = np.array(
        [
            geom_pos[0] + bbox_max[0] + ball_radius + contact_margin,
            geom_pos[1] + (bbox_min[1] + bbox_max[1]) / 2.0,
            geom_pos[2] + (bbox_min[2] + bbox_max[2]) / 2.0,
        ]
    ) + extra
    return offset, np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_euler_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """[roll, pitch, yaw] in degrees, intrinsic XYZ, from a wxyz quaternion."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = quat_wxyz
    rotation = Rotation.from_quat([x, y, z, w])  # scipy wants xyzw
    return rotation.as_euler("xyz", degrees=True)


@dataclass
class PalmFrame:
    position: np.ndarray
    quat_wxyz: np.ndarray
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    forward_vector: np.ndarray  # world-frame unit vector, local +x
    normal_vector: np.ndarray  # world-frame unit vector, local +z (convention, see module docstring)
    ball_position_local: np.ndarray  # ball position expressed IN the palm's own frame


def get_palm_frame(model, data, hand_body_id: int, ball_body_id: int) -> PalmFrame:
    position = data.xpos[hand_body_id].copy()
    quat = data.xquat[hand_body_id].copy()
    rot = data.xmat[hand_body_id].reshape(3, 3)
    euler = quat_to_euler_deg(quat)
    ball_world = data.xpos[ball_body_id]
    ball_local = rot.T @ (ball_world - position)
    return PalmFrame(
        position=position,
        quat_wxyz=quat,
        roll_deg=float(euler[0]),
        pitch_deg=float(euler[1]),
        yaw_deg=float(euler[2]),
        forward_vector=rot[:, 0].copy(),
        normal_vector=rot[:, 2].copy(),
        ball_position_local=ball_local,
    )


def print_hand_geometry_report(model, hand_body_id: int, ball_geom_id: int) -> None:
    """Item 4: print hand/ball geom names, contype/conaffinity, weld/joint info.
    Call once at startup (e.g. from the training script), not per env instance.
    """
    print("=" * 72)
    print("HAND / BALL GEOMETRY REPORT")
    print("=" * 72)
    hand_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, hand_body_id)
    print(f"hand body: '{hand_name}' (id={hand_body_id})")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != hand_body_id:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "(unnamed)"
        mesh_id = model.geom_dataid[g]
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else "(none)"
        print(
            f"  geom {g}: name={name} mesh={mesh_name} "
            f"contype={model.geom_contype[g]} conaffinity={model.geom_conaffinity[g]} "
            f"{'[COLLIDES]' if model.geom_contype[g] or model.geom_conaffinity[g] else '[VISUAL ONLY]'}"
        )
    hand_mesh_geom = find_hand_mesh_geom(model, hand_body_id)
    hand_mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hand_mesh_geom) if hand_mesh_geom >= 0 else None
    print(f"identified hand-mesh geom (furthest along local +x): id={hand_mesh_geom} name={hand_mesh_name}")
    ball_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ball_geom_id)
    print(
        f"ball geom: '{ball_name}' contype={model.geom_contype[ball_geom_id]} "
        f"conaffinity={model.geom_conaffinity[ball_geom_id]} radius={model.geom_size[ball_geom_id][0]:.4f}m"
    )
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_WELD:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "(unnamed)"
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj1id[e])
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.eq_obj2id[e])
        print(f"weld equality '{name}': body1={b1} body2={b2} relpose(pos,quat)={model.eq_data[e][3:10]}")
    print("Conclusion: hand mesh is VISUAL ONLY (no collision) -- the ball is placed by direct")
    print("qpos assignment + a weld equality constraint while held, never by hand-ball contact")
    print("physics. Release deactivates the weld (see G1FixedBodyThrowEnv.step); the ball has")
    print("its own free joint ('throw_ball_free') and is a true free body once released.")
    print("=" * 72)
