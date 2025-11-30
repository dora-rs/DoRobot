#!/usr/bin/env python3
"""
Standalone cloud training script for DoRobot.

This script uploads existing dataset from ~/DoRobot/dataset/ to cloud server
for training and downloads the trained model.

Usage:
    python scripts/cloud_train.py [OPTIONS]

Options:
    --dataset PATH      Path to dataset folder (default: auto-detect latest in ~/DoRobot/dataset/)
    --output PATH       Path to save trained model (default: ~/DoRobot/model)
    --api-url URL       API server URL (default: from env DOROBOT_API_URL or http://127.0.0.1:8000)
    --username USER     API username (default: from env DOROBOT_USERNAME or userb)
    --password PASS     API password (default: from env DOROBOT_PASSWORD or userb1234)
    --timeout MINUTES   Training timeout in minutes (default: 120)
    --list              List available datasets and exit

Examples:
    # Auto-detect latest dataset and train
    python scripts/cloud_train.py

    # Specify dataset path
    python scripts/cloud_train.py --dataset ~/DoRobot/dataset/20251130/experimental/so101-test

    # List available datasets
    python scripts/cloud_train.py --list

    # With custom API server
    DOROBOT_API_URL=http://192.168.0.12:8000 python scripts/cloud_train.py
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from operating_platform.core.cloud_train import run_cloud_training, log


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def find_datasets(base_path: Path) -> list:
    """Find all dataset folders under base_path"""
    datasets = []

    if not base_path.exists():
        return datasets

    # Look for datasets in the expected structure: {date}/experimental/{repo_id}
    for date_dir in sorted(base_path.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue

        experimental_dir = date_dir / "experimental"
        if not experimental_dir.exists():
            continue

        for repo_dir in sorted(experimental_dir.iterdir()):
            if not repo_dir.is_dir():
                continue

            # Check if it looks like a valid dataset (has meta/info.json)
            info_file = repo_dir / "meta" / "info.json"
            if info_file.exists():
                # Get modification time
                mtime = datetime.fromtimestamp(repo_dir.stat().st_mtime)
                datasets.append({
                    'path': repo_dir,
                    'date': date_dir.name,
                    'repo_id': repo_dir.name,
                    'modified': mtime,
                })

    return datasets


def list_datasets(base_path: Path):
    """List available datasets"""
    datasets = find_datasets(base_path)

    if not datasets:
        print(f"No datasets found in {base_path}")
        print("\nExpected structure:")
        print("  ~/DoRobot/dataset/{date}/experimental/{repo_id}/")
        print("  with meta/info.json file")
        return

    print(f"Available datasets in {base_path}:\n")
    print(f"{'#':<4} {'Date':<12} {'Repo ID':<20} {'Modified':<20} {'Path'}")
    print("-" * 100)

    for i, ds in enumerate(datasets, 1):
        print(f"{i:<4} {ds['date']:<12} {ds['repo_id']:<20} {ds['modified'].strftime('%Y-%m-%d %H:%M'):<20} {ds['path']}")

    print(f"\nTotal: {len(datasets)} dataset(s)")
    print("\nTo train a specific dataset:")
    print(f"  python scripts/cloud_train.py --dataset <PATH>")


def get_latest_dataset(base_path: Path) -> Path:
    """Get the most recently modified dataset"""
    datasets = find_datasets(base_path)

    if not datasets:
        return None

    # Sort by modification time and return the latest
    datasets.sort(key=lambda x: x['modified'], reverse=True)
    return datasets[0]['path']


def validate_dataset(dataset_path: Path) -> bool:
    """Validate that the path is a valid dataset"""
    if not dataset_path.exists():
        logging.error(f"Dataset path does not exist: {dataset_path}")
        return False

    if not dataset_path.is_dir():
        logging.error(f"Dataset path is not a directory: {dataset_path}")
        return False

    # Check for required files/folders
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        logging.error(f"Dataset missing meta/info.json: {dataset_path}")
        return False

    # Check for data folder
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        logging.error(f"Dataset missing data/ folder: {dataset_path}")
        return False

    # Check for images folder (cloud offload mode)
    images_dir = dataset_path / "images"
    if images_dir.exists():
        logging.info(f"Found images/ folder (cloud offload mode)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Upload dataset to cloud and train model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    default_base = Path.home() / "DoRobot" / "dataset"
    default_output = Path.home() / "DoRobot" / "model"

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help=f"Path to dataset folder (default: auto-detect latest in {default_base})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(default_output),
        help=f"Path to save trained model (default: {default_output})"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("DOROBOT_API_URL", "http://127.0.0.1:8000"),
        help="API server URL"
    )
    parser.add_argument(
        "--username", "-u",
        type=str,
        default=os.environ.get("DOROBOT_USERNAME", "userb"),
        help="API username"
    )
    parser.add_argument(
        "--password", "-p",
        type=str,
        default=os.environ.get("DOROBOT_PASSWORD", "userb1234"),
        help="API password"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=120,
        help="Training timeout in minutes (default: 120)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    setup_logging()

    # List mode
    if args.list:
        list_datasets(default_base)
        return 0

    # Determine dataset path
    if args.dataset:
        dataset_path = Path(args.dataset).expanduser().resolve()
    else:
        logging.info(f"Auto-detecting latest dataset in {default_base}...")
        dataset_path = get_latest_dataset(default_base)
        if dataset_path is None:
            logging.error(f"No datasets found in {default_base}")
            logging.info("Use --list to see available datasets or --dataset to specify path")
            return 1
        logging.info(f"Using latest dataset: {dataset_path}")

    # Validate dataset
    if not validate_dataset(dataset_path):
        return 1

    # Create output directory
    output_path = Path(args.output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("\n" + "=" * 60)
    print("  DoRobot Cloud Training")
    print("=" * 60)
    print(f"  Dataset:    {dataset_path}")
    print(f"  Output:     {output_path}")
    print(f"  API URL:    {args.api_url}")
    print(f"  Username:   {args.username}")
    print(f"  Timeout:    {args.timeout} minutes")
    print("=" * 60 + "\n")

    # Confirm before proceeding
    try:
        confirm = input("Start cloud training? [Y/n] ").strip().lower()
        if confirm and confirm != 'y':
            print("Cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return 0

    print()

    # Run cloud training
    try:
        success = run_cloud_training(
            dataset_path=str(dataset_path),
            model_output_path=str(output_path),
            api_url=args.api_url,
            username=args.username,
            password=args.password,
            timeout_minutes=args.timeout
        )

        if success:
            print("\n" + "=" * 60)
            print("  TRAINING COMPLETED SUCCESSFULLY!")
            print(f"  Model saved to: {output_path}")
            print("=" * 60 + "\n")
            return 0
        else:
            print("\n" + "=" * 60)
            print("  TRAINING FAILED")
            print("  Check the logs above for details")
            print("=" * 60 + "\n")
            return 1

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
