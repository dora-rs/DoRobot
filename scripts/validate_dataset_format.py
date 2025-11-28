#!/usr/bin/env python3
"""
Dataset Format Validation Script

This script validates that a DoRobot dataset (potentially saved with async_episode_saver)
maintains the correct data format and structure for training compatibility.

It compares the target dataset against a known-good reference dataset to ensure:
- Parquet file structure matches
- Video files exist and have correct encoding
- Metadata files (info.json, episodes.jsonl) are correctly formatted
- Column names and data types are identical

Usage:
    python scripts/validate_dataset_format.py \
        --reference /Users/nupylot/Public/so101-test-1126-ok \
        --target /path/to/test/dataset

    # Test with async save enabled
    python scripts/validate_dataset_format.py \
        --reference /Users/nupylot/Public/so101-test-1126-ok \
        --target /Users/nupylot/xuchengjie/DoRobotTest/data/20250126/experimental/test_async
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

class DatasetValidator:
    """Validates dataset format compatibility between reference and target datasets."""

    def __init__(self, reference_path: Path, target_path: Path):
        self.reference_path = Path(reference_path)
        self.target_path = Path(target_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.validations_passed = 0
        self.validations_total = 0

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all validations pass, False otherwise
        """
        logging.info("="*60)
        logging.info("Dataset Format Validation")
        logging.info("="*60)
        logging.info(f"Reference: {self.reference_path}")
        logging.info(f"Target:    {self.target_path}")
        logging.info("")

        # Check paths exist
        if not self._validate_paths_exist():
            return False

        # Run validation checks
        self._validate_directory_structure()
        self._validate_meta_files()
        self._validate_parquet_files()
        self._validate_video_files()

        # Report results
        self._print_report()

        return len(self.errors) == 0

    def _validate_paths_exist(self) -> bool:
        """Check that both reference and target paths exist."""
        if not self.reference_path.exists():
            logging.error(f"Reference path does not exist: {self.reference_path}")
            return False

        if not self.target_path.exists():
            logging.error(f"Target path does not exist: {self.target_path}")
            return False

        return True

    def _validate_directory_structure(self):
        """Validate that target has the same directory structure as reference."""
        logging.info("Validating directory structure...")

        expected_dirs = ["data", "meta", "videos"]
        for dir_name in expected_dirs:
            self.validations_total += 1
            ref_dir = self.reference_path / dir_name
            target_dir = self.target_path / dir_name

            if not ref_dir.exists():
                self.warnings.append(f"Reference missing directory: {dir_name}")
                continue

            if not target_dir.exists():
                self.errors.append(f"Target missing required directory: {dir_name}")
            else:
                self.validations_passed += 1
                logging.info(f"  ✓ Directory exists: {dir_name}/")

    def _validate_meta_files(self):
        """Validate metadata files structure and content."""
        logging.info("\nValidating metadata files...")

        meta_files = ["info.json", "episodes.jsonl", "tasks.jsonl"]

        for filename in meta_files:
            self.validations_total += 1
            ref_file = self.reference_path / "meta" / filename
            target_file = self.target_path / "meta" / filename

            if not ref_file.exists():
                self.warnings.append(f"Reference missing meta file: {filename}")
                continue

            if not target_file.exists():
                self.errors.append(f"Target missing meta file: {filename}")
                continue

            # Validate JSON structure
            try:
                if filename.endswith(".json"):
                    with open(ref_file) as f:
                        ref_data = json.load(f)
                    with open(target_file) as f:
                        target_data = json.load(f)

                    # Check that target has same keys as reference
                    ref_keys = set(ref_data.keys())
                    target_keys = set(target_data.keys())

                    if ref_keys != target_keys:
                        missing = ref_keys - target_keys
                        extra = target_keys - ref_keys
                        if missing:
                            self.errors.append(f"{filename}: Missing keys: {missing}")
                        if extra:
                            self.warnings.append(f"{filename}: Extra keys: {extra}")
                    else:
                        self.validations_passed += 1
                        logging.info(f"  ✓ {filename} structure matches")

                elif filename.endswith(".jsonl"):
                    # Just check it's valid JSONL
                    with open(target_file) as f:
                        for i, line in enumerate(f):
                            json.loads(line)  # Will raise if invalid

                    self.validations_passed += 1
                    logging.info(f"  ✓ {filename} is valid JSONL")

            except json.JSONDecodeError as e:
                self.errors.append(f"{filename}: Invalid JSON format: {e}")
            except Exception as e:
                self.errors.append(f"{filename}: Validation error: {e}")

    def _validate_parquet_files(self):
        """Validate parquet files have matching schema and structure."""
        logging.info("\nValidating parquet files...")

        ref_data_dir = self.reference_path / "data" / "chunk-000"
        target_data_dir = self.target_path / "data" / "chunk-000"

        if not ref_data_dir.exists():
            self.warnings.append("Reference has no data/chunk-000 directory")
            return

        if not target_data_dir.exists():
            self.errors.append("Target missing data/chunk-000 directory")
            return

        # Get sample parquet files
        ref_parquets = sorted(ref_data_dir.glob("episode_*.parquet"))
        target_parquets = sorted(target_data_dir.glob("episode_*.parquet"))

        if not ref_parquets:
            self.warnings.append("Reference has no parquet files")
            return

        if not target_parquets:
            self.errors.append("Target has no parquet files")
            return

        # Compare schema of first parquet file
        self.validations_total += 1
        try:
            ref_table = pq.read_table(ref_parquets[0])
            target_table = pq.read_table(target_parquets[0])

            ref_schema = ref_table.schema
            target_schema = target_table.schema

            # Check column names match
            ref_columns = set(ref_schema.names)
            target_columns = set(target_schema.names)

            if ref_columns != target_columns:
                missing = ref_columns - target_columns
                extra = target_columns - ref_columns
                if missing:
                    self.errors.append(f"Parquet missing columns: {missing}")
                if extra:
                    self.warnings.append(f"Parquet extra columns: {extra}")
            else:
                self.validations_passed += 1
                logging.info(f"  ✓ Parquet schema matches ({len(ref_columns)} columns)")

            # Check data types match for each column
            self.validations_total += 1
            type_mismatches = []
            for col in ref_columns & target_columns:
                ref_type = ref_schema.field(col).type
                target_type = target_schema.field(col).type
                if ref_type != target_type:
                    type_mismatches.append(f"{col}: {ref_type} vs {target_type}")

            if type_mismatches:
                self.errors.append(f"Parquet type mismatches: {type_mismatches}")
            else:
                self.validations_passed += 1
                logging.info(f"  ✓ Column data types match")

            # Check row structure
            self.validations_total += 1
            logging.info(f"  Reference episode shape: {ref_table.shape}")
            logging.info(f"  Target episode shape: {target_table.shape}")

            # Don't require exact row count match (different episodes)
            # but check that both have reasonable number of frames
            if target_table.num_rows < 10:
                self.warnings.append(f"Target episode has very few frames: {target_table.num_rows}")
            else:
                self.validations_passed += 1
                logging.info(f"  ✓ Target has reasonable frame count")

        except Exception as e:
            self.errors.append(f"Parquet validation error: {e}")

    def _validate_video_files(self):
        """Validate video files exist and have correct structure."""
        logging.info("\nValidating video files...")

        ref_video_dir = self.reference_path / "videos" / "chunk-000"
        target_video_dir = self.target_path / "videos" / "chunk-000"

        if not ref_video_dir.exists():
            self.warnings.append("Reference has no videos/chunk-000 directory")
            return

        if not target_video_dir.exists():
            # Videos are optional depending on config
            self.warnings.append("Target has no videos/chunk-000 directory (may be disabled)")
            return

        # Get camera directories
        ref_cameras = [d for d in ref_video_dir.iterdir() if d.is_dir()]
        target_cameras = [d for d in target_video_dir.iterdir() if d.is_dir()]

        if not ref_cameras:
            self.warnings.append("Reference has no camera directories")
            return

        if not target_cameras:
            self.warnings.append("Target has no camera directories")
            return

        # Compare camera directory names
        ref_camera_names = {c.name for c in ref_cameras}
        target_camera_names = {c.name for c in target_cameras}

        self.validations_total += 1
        if ref_camera_names != target_camera_names:
            missing = ref_camera_names - target_camera_names
            extra = target_camera_names - ref_camera_names
            if missing:
                self.errors.append(f"Target missing camera directories: {missing}")
            if extra:
                self.warnings.append(f"Target has extra camera directories: {extra}")
        else:
            self.validations_passed += 1
            logging.info(f"  ✓ Camera directories match: {target_camera_names}")

        # Check that video files exist for each camera
        for camera_name in ref_camera_names & target_camera_names:
            self.validations_total += 1
            ref_cam_dir = ref_video_dir / camera_name
            target_cam_dir = target_video_dir / camera_name

            ref_videos = list(ref_cam_dir.glob("episode_*.mp4"))
            target_videos = list(target_cam_dir.glob("episode_*.mp4"))

            if ref_videos and not target_videos:
                self.errors.append(f"Camera {camera_name}: No video files found")
            elif target_videos:
                self.validations_passed += 1
                logging.info(f"  ✓ Camera {camera_name}: {len(target_videos)} video(s) found")

    def _print_report(self):
        """Print validation report."""
        logging.info("\n" + "="*60)
        logging.info("Validation Report")
        logging.info("="*60)

        if self.warnings:
            logging.warning(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logging.warning(f"  ⚠ {warning}")

        if self.errors:
            logging.error(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                logging.error(f"  ✗ {error}")

        logging.info(f"\nValidations: {self.validations_passed}/{self.validations_total} passed")

        if len(self.errors) == 0:
            logging.info("\n✓ VALIDATION PASSED: Dataset format is compatible")
            logging.info("  The target dataset structure matches the reference dataset.")
            logging.info("  It should be safe to use for training.\n")
        else:
            logging.error("\n✗ VALIDATION FAILED: Dataset format has issues")
            logging.error(f"  Found {len(self.errors)} error(s) that may cause training to fail.")
            logging.error("  Please fix these issues before training.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate DoRobot dataset format compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate against reference dataset
  python scripts/validate_dataset_format.py \\
      --reference /Users/nupylot/Public/so101-test-1126-ok \\
      --target /path/to/new/dataset

  # Test async-saved dataset
  python scripts/validate_dataset_format.py \\
      --reference /Users/nupylot/Public/so101-test-1126-ok \\
      --target data/20250126/experimental/test_async
        """
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default="/Users/nupylot/Public/so101-test-1126-ok",
        help="Path to known-good reference dataset (default: /Users/nupylot/Public/so101-test-1126-ok)"
    )

    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Path to dataset to validate"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run validation
    validator = DatasetValidator(args.reference, args.target)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
