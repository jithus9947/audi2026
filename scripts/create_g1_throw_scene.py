#!/usr/bin/env python3
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[1]

BALL_AND_TARGET_XML = """
    <body name="throw_ball" pos="0.31 -0.15 0.89">
      <freejoint name="throw_ball_free"/>
      <geom name="throw_ball_geom" type="sphere" size="0.04" mass="0.05" rgba="0.9 0.2 0.1 1"/>
    </body>

    <body name="throw_target" pos="0.45 0.0 0.01">
      <geom name="throw_target_geom" type="cylinder" size="0.18 0.003"
            contype="0" conaffinity="0" rgba="0.1 0.8 0.2 0.45"/>
    </body>
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand-body",
        required=True,
        help="Right hand or wrist body/frame name from inspect_g1_bodies.py",
    )
    parser.add_argument(
        "--hand-offset",
        type=float,
        nargs=3,
        default=(0.16, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Ball position relative to --hand-body in metres (default: 0.16 0 0, held clear of the palm and torso).",
    )
    parser.add_argument(
        "--source",
        default=str(ROOT / "assets" / "unitree_g1" / "scene.xml"),
        help="Original Unitree G1 rigid five-finger visual-hand scene XML",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "assets" / "unitree_g1" / "scene_throw.xml"),
        help="Generated throwing scene. Keep it inside assets/unitree_g1/ to preserve relative mesh paths.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(
            f"Source scene not found: {source}\n"
            "First copy the Unitree G1 model into assets/unitree_g1/."
        )

    text = source.read_text(encoding="utf-8")

    if "throw_ball" in text or "hold_throw_ball" in text:
        raise RuntimeError(
            "The source scene already seems to contain throw_ball or hold_throw_ball. "
            "Use the original clean assets/unitree_g1/scene.xml as input."
        )

    if "</worldbody>" not in text:
        raise RuntimeError("Could not find </worldbody> in source scene.")

    if "</mujoco>" not in text:
        raise RuntimeError("Could not find </mujoco> in source scene.")

    text = text.replace("</worldbody>", BALL_AND_TARGET_XML + "\n  </worldbody>", 1)

    hand_offset = " ".join(f"{value:g}" for value in args.hand_offset)
    equality_xml = f"""
  <equality>
    <weld name="hold_throw_ball"
          body1="{args.hand_body}"
          body2="throw_ball"
          relpose="{hand_offset} 1 0 0 0"
          active="true"
          solref="0.002 1"
          solimp="0.95 0.99 0.001"/>
  </equality>
"""

    text = text.replace("</mujoco>", equality_xml + "\n</mujoco>", 1)

    output.write_text(text, encoding="utf-8")

    print(f"Wrote: {output}")
    print()
    print("Test with:")
    print(
        "python3 -c \"import mujoco; "
        "mujoco.MjModel.from_xml_path('assets/unitree_g1/scene_throw.xml'); "
        "print('G1 throw XML OK')\""
    )


if __name__ == "__main__":
    main()
