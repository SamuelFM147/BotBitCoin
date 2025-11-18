import sys
import importlib

def get_versions():
    v = {}
    libs = [
        ("numpy", "__version__"),
        ("pandas", "__version__"),
        ("torch", "__version__"),
        ("stable_baselines3", "__version__"),
        ("gymnasium", "__version__"),
        ("scikit_learn", "__version__"),
        ("ta", "__version__"),
        ("requests", "__version__"),
        ("pyyaml", "__version__"),
    ]
    for name, attr in libs:
        try:
            m = importlib.import_module(name)
            v[name] = getattr(m, attr)
        except Exception:
            v[name] = None
    v["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return v

def is_python_supported():
    return (sys.version_info.major == 3) and (11 <= sys.version_info.minor < 13)

if __name__ == "__main__":
    print(get_versions())