# main.py
import argparse
from src.runner import APEExperimentRunner

def main():
    parser = argparse.ArgumentParser(description="APE Project Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml", 
        help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    runner = APEExperimentRunner(args.config)
    runner.run_all()

if __name__ == "__main__":
    main()