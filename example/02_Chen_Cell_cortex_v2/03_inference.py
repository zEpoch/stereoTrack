from __future__ import annotations

import runpy
import sys
from pathlib import Path


WORK_DIR = Path("/home/share/huadjyin/home/zhoutao3/tracks/stereoTrack")
DEFAULT_CONFIG = WORK_DIR / "config/02_config_Chen_Cell_cortex_v2.yaml"
GENERIC_INFERENCE = WORK_DIR / "example/06_mouse_embryo_e115/03_inference.py"


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", str(DEFAULT_CONFIG)])
    runpy.run_path(str(GENERIC_INFERENCE), run_name="__main__")


if __name__ == "__main__":
    main()

