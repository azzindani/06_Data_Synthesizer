"""Output handlers for saving synthesized data."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.logger import get_logger


class LocalOutputHandler:
    """Handler for saving data to local filesystem."""

    def __init__(self, base_path: str, config: Any = None):
        """Initialize local output handler.

        Args:
            base_path: Base directory for output
            config: Optional configuration object
        """
        self.logger = get_logger(__name__)
        self.base_path = Path(base_path)
        self.config = config

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.base_path}")

    def save_batch(self, data: List[Dict[str, Any]], filename: str,
                   append: bool = True) -> str:
        """Save a batch of data to file.

        Args:
            data: List of data items to save
            filename: Output filename (without extension)
            append: Whether to append to existing file

        Returns:
            Path to saved file
        """
        filepath = self.base_path / f"{filename}.jsonl"

        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        self.logger.debug(f"Saved {len(data)} items to {filepath}")
        return str(filepath)

    def save_json(self, data: Any, filename: str) -> str:
        """Save data as JSON file.

        Args:
            data: Data to save
            filename: Output filename (without extension)

        Returns:
            Path to saved file
        """
        filepath = self.base_path / f"{filename}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"Saved JSON to {filepath}")
        return str(filepath)

    def save_checkpoint(self, state: Dict[str, Any], name: str = "checkpoint") -> str:
        """Save checkpoint state.

        Args:
            state: State dictionary to save
            name: Checkpoint name

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = self.base_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = checkpoint_dir / f"{name}_{timestamp}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved checkpoint: {filepath}")
        return str(filepath)

    def load_checkpoint(self, name: str = "checkpoint") -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint.

        Args:
            name: Checkpoint name prefix

        Returns:
            Checkpoint state or None
        """
        checkpoint_dir = self.base_path / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        # Find most recent checkpoint
        checkpoints = sorted(
            checkpoint_dir.glob(f"{name}_*.json"),
            reverse=True
        )

        if not checkpoints:
            return None

        latest = checkpoints[0]
        with open(latest, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.logger.info(f"Loaded checkpoint: {latest}")
        return state

    def get_output_files(self) -> List[str]:
        """List all output files.

        Returns:
            List of file paths
        """
        files = []
        for pattern in ['*.jsonl', '*.json']:
            files.extend(str(f) for f in self.base_path.glob(pattern))
        return sorted(files)

    def get_item_count(self, filename: str) -> int:
        """Get count of items in a JSONL file.

        Args:
            filename: Filename to count

        Returns:
            Number of items
        """
        filepath = self.base_path / f"{filename}.jsonl"
        if not filepath.exists():
            return 0

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count

    def merge_outputs(self, output_files: List[str], merged_name: str) -> str:
        """Merge multiple JSONL files into one.

        Args:
            output_files: List of filenames to merge
            merged_name: Name for merged file

        Returns:
            Path to merged file
        """
        merged_path = self.base_path / f"{merged_name}.jsonl"

        with open(merged_path, 'w', encoding='utf-8') as outfile:
            for filename in output_files:
                filepath = self.base_path / f"{filename}.jsonl"
                if filepath.exists():
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)

        self.logger.info(f"Merged {len(output_files)} files to {merged_path}")
        return str(merged_path)

    def export_to_csv(self, filename: str, fields: List[str] = None) -> str:
        """Export JSONL to CSV format.

        Args:
            filename: Input filename (without extension)
            fields: Fields to include (None for all)

        Returns:
            Path to CSV file
        """
        import csv

        jsonl_path = self.base_path / f"{filename}.jsonl"
        csv_path = self.base_path / f"{filename}.csv"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"File not found: {jsonl_path}")

        # Read all items
        items = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                items.append(json.loads(line))

        if not items:
            return str(csv_path)

        # Determine fields
        if fields is None:
            fields = list(items[0].keys())

        # Write CSV
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(items)

        self.logger.info(f"Exported to CSV: {csv_path}")
        return str(csv_path)


class HuggingFaceOutputHandler:
    """Handler for saving data to HuggingFace Hub."""

    def __init__(self, repo_id: str, token: str = None, config: Any = None):
        """Initialize HuggingFace output handler.

        Args:
            repo_id: HuggingFace repository ID (user/dataset)
            token: HuggingFace token
            config: Optional configuration object
        """
        self.logger = get_logger(__name__)
        self.repo_id = repo_id
        self.token = token or os.getenv('HF_TOKEN')
        self.config = config

        self._api = None

    def _get_api(self):
        """Get HuggingFace API client."""
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def upload_file(self, local_path: str, repo_path: str = None) -> str:
        """Upload file to HuggingFace Hub.

        Args:
            local_path: Local file path
            repo_path: Path in repository (default: filename)

        Returns:
            URL of uploaded file
        """
        if repo_path is None:
            repo_path = Path(local_path).name

        try:
            api = self._get_api()
            url = api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            self.logger.info(f"Uploaded to HuggingFace: {url}")
            return url

        except Exception as e:
            self.logger.error(f"HuggingFace upload failed: {e}")
            raise

    def upload_folder(self, local_folder: str, repo_folder: str = None) -> str:
        """Upload folder to HuggingFace Hub.

        Args:
            local_folder: Local folder path
            repo_folder: Folder path in repository

        Returns:
            Repository URL
        """
        try:
            api = self._get_api()
            url = api.upload_folder(
                folder_path=local_folder,
                path_in_repo=repo_folder or "",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            self.logger.info(f"Uploaded folder to HuggingFace: {url}")
            return url

        except Exception as e:
            self.logger.error(f"HuggingFace folder upload failed: {e}")
            raise


class OutputManager:
    """Unified output manager supporting multiple destinations."""

    def __init__(self, config: Any):
        """Initialize output manager.

        Args:
            config: Configuration object with output settings
        """
        self.logger = get_logger(__name__)
        self.config = config

        self._local = None
        self._hf = None

        # Initialize handlers based on config
        output_config = getattr(config, 'output', None)
        if output_config:
            output_type = getattr(output_config, 'type', 'local')

            if output_type in ['local', 'both']:
                local_path = getattr(output_config, 'local_path', './output')
                self._local = LocalOutputHandler(local_path, config)

            if output_type in ['huggingface', 'both']:
                hf_repo = getattr(output_config, 'hf_repo', '')
                hf_token = getattr(output_config, 'hf_token', '')
                if hf_repo:
                    self._hf = HuggingFaceOutputHandler(hf_repo, hf_token, config)

    def save(self, data: List[Dict[str, Any]], filename: str) -> Dict[str, str]:
        """Save data to all configured outputs.

        Args:
            data: Data to save
            filename: Base filename

        Returns:
            Dict with paths/URLs for each destination
        """
        results = {}

        if self._local:
            results['local'] = self._local.save_batch(data, filename)

        if self._hf and self._local:
            # Upload local file to HF
            local_path = results.get('local')
            if local_path:
                results['huggingface'] = self._hf.upload_file(local_path)

        return results


if __name__ == "__main__":
    print("=" * 60)
    print("OUTPUT HANDLER TEST")
    print("=" * 60)

    # Test 1: Local handler
    handler = LocalOutputHandler("./test_output")

    # Save batch
    test_data = [
        {"question": "Test Q1", "answer": "Test A1"},
        {"question": "Test Q2", "answer": "Test A2"},
    ]
    path = handler.save_batch(test_data, "test_qa")
    print(f"  ✓ Saved batch: {path}")

    # Save JSON
    json_path = handler.save_json({"test": "data"}, "test_config")
    print(f"  ✓ Saved JSON: {json_path}")

    # Get item count
    count = handler.get_item_count("test_qa")
    print(f"  ✓ Item count: {count}")

    # Save checkpoint
    checkpoint = handler.save_checkpoint({"processed": 10})
    print(f"  ✓ Saved checkpoint: {checkpoint}")

    # Load checkpoint
    loaded = handler.load_checkpoint()
    print(f"  ✓ Loaded checkpoint: {loaded}")

    # List files
    files = handler.get_output_files()
    print(f"  ✓ Output files: {len(files)}")

    # Cleanup
    import shutil
    shutil.rmtree("./test_output")
    print("  ✓ Cleaned up test files")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
