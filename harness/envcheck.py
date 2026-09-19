from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import sys


def _fit3d_status() -> dict:
    value=os.environ.get("FIT3D_ROOT")
    if not value:
        return {"configured":False,"root":None,"exists":False}
    p=Path(value).expanduser()
    if not (p/"train").exists() and (p/"fit3d"/"train").exists():
        p=p/"fit3d"
    return {"configured":True,"root":str(p),"exists":p.exists(),"train_exists":(p/"train").exists(),"test_exists":(p/"test").exists()}


def environment_report() -> dict:
    modules=["numpy","pandas","cv2","scipy","trimesh","pytest","hypothesis","pydantic","pyarrow","mediapipe"]
    bins=["python","git","ffmpeg","blender","java","adb"]
    return {
        "platform":platform.platform(),
        "python":sys.version.split()[0],
        "modules":{m:importlib.util.find_spec(m) is not None for m in modules},
        "binaries":{b:shutil.which(b) for b in bins},
        "fit3d":_fit3d_status(),
    }


def print_environment_report() -> None:
    print(json.dumps(environment_report(),indent=2))
