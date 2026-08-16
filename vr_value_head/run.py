"""One-glance runner for the value-head pipeline.

All three stages run in ONE venv — /opt/venv/egoverse already has zarr AND the full torch stack
(torch 2.7.1+cu128, transformers, sm_120), so it reads egoverse frames and runs the frozen
Cosmos-Reason2-2B backbone. This prints the commands; `bashscripts/pretrain_value_head/run.sh`
runs them with staging/logging.

    python run.py            # print the pipeline
"""

VENV = "/opt/venv/egoverse"
STAGES = [
    ("1. preprocess  (reads zarr frames, no backbone)",
     f"{VENV}/bin/python build_labels.py --episodes 200 --stride 10 --arm subgoal --neg 1"),
    ("2. extract     (frozen Cosmos-Reason2-2B backbone, needs a GPU)",
     f"{VENV}/bin/python extract_features.py --cache cache --batch-size 16"),
    ("3. train+eval  (MLP on cached features, seconds, CPU ok)",
     f"{VENV}/bin/python train.py --cache cache --pool mean_last --time-warp"),
]

if __name__ == "__main__":
    print("Value-head pipeline (original egoverse → GR00T N1.7 features → progress head)")
    print(f"single venv: {VENV}\n")
    for title, cmd in STAGES:
        print(f"# {title}\n{cmd}\n")
    print("Sweep pooling/arm without re-extracting: rerun stage 3 with --pool {last_last,mean_mid};\n"
          "rerun stage 1 with --arm full for the faithful-span comparison (re-extract needed there).")
