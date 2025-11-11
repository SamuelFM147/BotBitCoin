import torch

def main():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    try:
        print(f"CUDA version (torch): {torch.version.cuda}")
    except Exception:
        print("CUDA version (torch): unknown")
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "unknown"
        print(f"CUDA device: {device_name}")
    else:
        print("CUDA device: cpu")

if __name__ == "__main__":
    main()