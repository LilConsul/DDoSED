from src.train import build_model_registry

if __name__ == "__main__":
    print(sorted(build_model_registry().keys()))
