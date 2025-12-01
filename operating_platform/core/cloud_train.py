#!/usr/bin/env python3
"""
Cloud training module for DoRobot.
Uploads dataset to cloud server for training and downloads the trained model.

This is integrated with main.py to provide a 't' key option for cloud-based training,
which is useful for edge devices (like Orange Pi) where local video encoding is slow.

Uses paramiko for SSH/SFTP operations (no external sshpass dependency).
"""

import requests
import base64
import os
import time
import threading
import logging
from pathlib import Path

# Default API configuration - can be overridden via environment variables
DEFAULT_API_BASE_URL = os.environ.get("DOROBOT_API_URL", "http://127.0.0.1:8000")
DEFAULT_USERNAME = os.environ.get("DOROBOT_USERNAME", "userb")
DEFAULT_PASSWORD = os.environ.get("DOROBOT_PASSWORD", "userb1234")


def log(message):
    """Print timestamped log messages"""
    logging.info(f"[CloudTrain] {message}")


class CloudTrainer:
    """
    Handles cloud-based training workflow:
    1. Login to API server
    2. Start transaction with cloud_offload mode
    3. Upload dataset via SFTP (raw images, no videos)
    4. Wait for training to complete
    5. Download trained model
    """

    def __init__(self, api_base_url: str = None, username: str = None, password: str = None):
        self.api_base_url = api_base_url or DEFAULT_API_BASE_URL
        self.username = username or DEFAULT_USERNAME
        self.password = password or DEFAULT_PASSWORD
        self.token = None
        self.transaction_id = None
        self.ssh_info = None
        self.remote_path = None
        self._ssh_client = None

    def login(self) -> bool:
        """Login and get authentication token"""
        log("Logging in to API server...")
        try:
            response = requests.post(
                f"{self.api_base_url}/login",
                json={"username": self.username, "password": self.password},
                timeout=30
            )
            response.raise_for_status()
            login_data = response.json()
            self.token = login_data["access_token"]
            log(f"Logged in as: {login_data['username']}")
            return True
        except Exception as e:
            log(f"Login failed: {e}")
            return False

    def start_transaction(self, cloud_offload: bool = True) -> bool:
        """Start a new training transaction"""
        log("Starting training transaction...")

        training_config = {
            "model_type": "act",
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_docker": False,
            "cloud_offload": cloud_offload,
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/transactions/start",
                json={"training_config": training_config},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=30
            )
            response.raise_for_status()
            transaction_data = response.json()

            self.transaction_id = transaction_data["transaction_id"]
            log(f"Transaction started: {self.transaction_id}")

            # Check SSH credentials
            if not all(k in transaction_data for k in ["ssh_host", "ssh_username", "ssh_password", "ssh_port"]):
                log("SSH credentials not available")
                return False

            # Setup SSH info
            ssh_password = base64.b64decode(transaction_data["ssh_password"]).decode('utf-8')
            self.ssh_info = {
                'host': transaction_data["ssh_host"],
                'username': transaction_data["ssh_username"],
                'password': ssh_password,
                'port': int(transaction_data["ssh_port"])
            }

            # Calculate remote path
            s3_data_path = transaction_data["s3_data_path"]
            self.remote_path = f"/root/{s3_data_path}"
            log(f"Remote path: {self.remote_path}")

            return True
        except Exception as e:
            log(f"Failed to start transaction: {e}")
            return False

    def _get_ssh_client(self):
        """Get or create SSH client using paramiko"""
        import paramiko

        if self._ssh_client is not None:
            return self._ssh_client

        log(f"Connecting to {self.ssh_info['username']}@{self.ssh_info['host']}:{self.ssh_info['port']}...")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.ssh_info['host'],
            port=self.ssh_info['port'],
            username=self.ssh_info['username'],
            password=self.ssh_info['password'],
            timeout=30,
            allow_agent=False,
            look_for_keys=False
        )
        self._ssh_client = client
        return client

    def _close_ssh_client(self):
        """Close SSH client"""
        if self._ssh_client is not None:
            try:
                self._ssh_client.close()
            except:
                pass
            self._ssh_client = None

    def test_ssh_connection(self) -> bool:
        """Test SSH connection using paramiko"""
        log("Testing SSH connection...")
        try:
            client = self._get_ssh_client()
            stdin, stdout, stderr = client.exec_command("echo 'SSH OK'")
            result = stdout.read().decode().strip()
            if "SSH OK" in result:
                log("SSH connection successful")
                return True
            else:
                log(f"SSH connection failed: {stderr.read().decode()}")
                return False
        except Exception as e:
            log(f"SSH test error: {e}")
            self._close_ssh_client()
            return False

    def create_remote_directory(self) -> bool:
        """Create remote directory using paramiko"""
        log(f"Creating remote directory: {self.remote_path}")
        try:
            client = self._get_ssh_client()
            stdin, stdout, stderr = client.exec_command(f"mkdir -p {self.remote_path}")
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                log("Remote directory created")
                return True
            else:
                log(f"Failed to create directory: {stderr.read().decode()}")
                return False
        except Exception as e:
            log(f"Failed to create remote directory: {e}")
            return False

    def upload_dataset(self, local_path: str, progress_callback=None) -> bool:
        """Upload dataset using SFTP (paramiko)"""
        log(f"Uploading dataset from: {local_path}")
        log(f"To: {self.ssh_info['username']}@{self.ssh_info['host']}:{self.remote_path}")

        try:
            client = self._get_ssh_client()
            sftp = client.open_sftp()

            local_path = Path(local_path)
            total_files = 0
            uploaded_files = 0

            # Count total files first
            for root, dirs, files in os.walk(local_path):
                total_files += len(files)

            log(f"Total files to upload: {total_files}")

            # Upload files recursively
            for root, dirs, files in os.walk(local_path):
                # Calculate relative path
                rel_root = Path(root).relative_to(local_path)
                remote_dir = f"{self.remote_path}/{rel_root}" if str(rel_root) != "." else self.remote_path

                # Create remote directories
                self._sftp_makedirs(sftp, remote_dir)

                # Upload files
                for filename in files:
                    local_file = Path(root) / filename
                    remote_file = f"{remote_dir}/{filename}"

                    try:
                        sftp.put(str(local_file), remote_file)
                        uploaded_files += 1

                        if uploaded_files % 100 == 0 or uploaded_files == total_files:
                            progress = (uploaded_files / total_files) * 100
                            log(f"Upload progress: {uploaded_files}/{total_files} ({progress:.1f}%)")
                            if progress_callback:
                                progress_callback(f"{uploaded_files}/{total_files} files")
                    except Exception as e:
                        log(f"Failed to upload {local_file}: {e}")
                        # Continue with other files

            sftp.close()
            log(f"Upload completed: {uploaded_files}/{total_files} files")
            return uploaded_files > 0

        except Exception as e:
            log(f"Upload error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _sftp_makedirs(self, sftp, remote_dir: str):
        """Recursively create remote directories"""
        dirs_to_create = []
        current = remote_dir

        while current and current != "/":
            try:
                sftp.stat(current)
                break  # Directory exists
            except FileNotFoundError:
                dirs_to_create.append(current)
                current = str(Path(current).parent)

        # Create directories from parent to child
        for d in reversed(dirs_to_create):
            try:
                sftp.mkdir(d)
            except:
                pass  # Directory might already exist

    def mark_upload_complete(self) -> bool:
        """Mark upload as complete to trigger training"""
        log("Marking upload complete...")
        try:
            response = requests.post(
                f"{self.api_base_url}/transactions/{self.transaction_id}/upload-complete",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=30
            )
            response.raise_for_status()
            log("Upload marked complete - training starting")
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                # Token expired, try to refresh
                if self.login():
                    return self.mark_upload_complete()
            log(f"Failed to mark upload complete: {e}")
            return False
        except Exception as e:
            log(f"Error marking upload complete: {e}")
            return False

    def poll_training_status(self, timeout_minutes: int = 60, poll_interval: int = 10,
                            status_callback=None) -> tuple:
        """
        Poll training status until completion.
        Returns (success: bool, model_path: str or None)
        """
        log(f"Monitoring training (timeout: {timeout_minutes} min)...")

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while (time.time() - start_time) < timeout_seconds:
            try:
                headers = {"Authorization": f"Bearer {self.token}"}
                resp = requests.get(
                    f"{self.api_base_url}/transactions/{self.transaction_id}/status",
                    headers=headers,
                    timeout=30
                )

                if resp.status_code == 401:
                    if self.login():
                        continue
                    return False, None

                if resp.status_code == 200:
                    data = resp.json()
                    status = str(data.get("status", "UNKNOWN")).upper()
                    progress = data.get("training_progress", "")

                    if status_callback:
                        status_callback(status, progress)
                    else:
                        log(f"Status: {status}, Progress: {progress}")

                    if status == "COMPLETED":
                        model_path = data.get("model_path")
                        log(f"Training completed! Model path: {model_path}")
                        return True, model_path
                    elif status in ("FAILED", "ERROR", "CANCELLED"):
                        error = data.get("error_message", "Unknown error")
                        log(f"Training failed: {error}")
                        return False, None

                time.sleep(poll_interval)

            except Exception as e:
                log(f"Status check error: {e}")
                time.sleep(poll_interval)

        log("Training monitoring timeout")
        return False, None

    def download_model(self, remote_model_path: str, local_output_path: str) -> bool:
        """Download trained model via SFTP (paramiko)"""
        log(f"Downloading model from: {remote_model_path}")
        log(f"To: {local_output_path}")

        try:
            Path(local_output_path).mkdir(parents=True, exist_ok=True)

            client = self._get_ssh_client()
            sftp = client.open_sftp()

            downloaded_files = 0

            def download_recursive(remote_dir, local_dir):
                nonlocal downloaded_files
                Path(local_dir).mkdir(parents=True, exist_ok=True)

                try:
                    for entry in sftp.listdir_attr(remote_dir):
                        remote_path = f"{remote_dir}/{entry.filename}"
                        local_path = Path(local_dir) / entry.filename

                        if entry.st_mode & 0o40000:  # Is directory
                            download_recursive(remote_path, str(local_path))
                        else:
                            sftp.get(remote_path, str(local_path))
                            downloaded_files += 1
                            if downloaded_files % 10 == 0:
                                log(f"Downloaded {downloaded_files} files...")
                except Exception as e:
                    log(f"Error downloading from {remote_dir}: {e}")

            download_recursive(remote_model_path, local_output_path)
            sftp.close()

            log(f"Model download completed: {downloaded_files} files")
            return downloaded_files > 0

        except Exception as e:
            log(f"Download error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def modify_config_device(self, local_model_path: str, to_device: str = "npu") -> bool:
        """Modify config.json device setting for local inference"""
        import json
        config_path = Path(local_model_path) / "config.json"

        if not config_path.exists():
            log(f"config.json not found at {config_path}")
            return False

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            original_device = config.get('device', 'unknown')
            config['device'] = to_device

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            log(f"Updated device: {original_device} -> {to_device}")
            return True
        except Exception as e:
            log(f"Failed to modify config: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        self._close_ssh_client()


def run_cloud_training(dataset_path: str, model_output_path: str,
                       api_url: str = None, username: str = None, password: str = None,
                       timeout_minutes: int = 60,
                       status_callback=None) -> bool:
    """
    Convenience function to run the full cloud training workflow.

    Args:
        dataset_path: Local path to dataset (with images/, data/, meta/ folders)
        model_output_path: Local path to save downloaded model
        api_url: API server URL (default from env or http://127.0.0.1:8000)
        username: API username
        password: API password
        timeout_minutes: Training timeout in minutes
        status_callback: Optional callback(status, progress) for status updates

    Returns:
        True if training completed and model downloaded successfully
    """
    trainer = CloudTrainer(api_base_url=api_url, username=username, password=password)

    try:
        # Step 1: Login
        if not trainer.login():
            return False

        # Step 2: Start transaction
        if not trainer.start_transaction(cloud_offload=True):
            return False

        # Step 3: Test SSH
        if not trainer.test_ssh_connection():
            return False

        # Step 4: Create remote directory
        if not trainer.create_remote_directory():
            return False

        # Step 5: Upload dataset
        if not trainer.upload_dataset(dataset_path):
            return False

        # Step 6: Mark upload complete
        if not trainer.mark_upload_complete():
            return False

        # Step 7: Poll training status
        success, model_path = trainer.poll_training_status(
            timeout_minutes=timeout_minutes,
            status_callback=status_callback
        )

        if not success or not model_path:
            log("Training did not complete successfully")
            return False

        # Step 8: Download model
        if not trainer.download_model(model_path, model_output_path):
            return False

        # Step 9: Modify config for local device
        trainer.modify_config_device(model_output_path, to_device="npu")

        log(f"Cloud training complete! Model saved to: {model_output_path}")
        return True

    finally:
        trainer.cleanup()


class CloudTrainingThread(threading.Thread):
    """
    Background thread for cloud training.
    Allows the main program to continue while training runs in the background.
    """

    def __init__(self, dataset_path: str, model_output_path: str,
                 api_url: str = None, username: str = None, password: str = None,
                 timeout_minutes: int = 60):
        super().__init__(daemon=True)
        self.dataset_path = dataset_path
        self.model_output_path = model_output_path
        self.api_url = api_url
        self.username = username
        self.password = password
        self.timeout_minutes = timeout_minutes

        self.success = False
        self.error_message = None
        self.current_status = "INITIALIZING"
        self.current_progress = ""
        self.completed = threading.Event()

    def run(self):
        try:
            def status_callback(status, progress):
                self.current_status = status
                self.current_progress = progress

            self.success = run_cloud_training(
                dataset_path=self.dataset_path,
                model_output_path=self.model_output_path,
                api_url=self.api_url,
                username=self.username,
                password=self.password,
                timeout_minutes=self.timeout_minutes,
                status_callback=status_callback
            )

            if self.success:
                self.current_status = "COMPLETED"
            else:
                self.current_status = "FAILED"

        except Exception as e:
            self.success = False
            self.error_message = str(e)
            self.current_status = "ERROR"
            log(f"Cloud training thread error: {e}")
        finally:
            self.completed.set()

    def wait_for_completion(self, timeout: float = None) -> bool:
        """Wait for training to complete. Returns True if completed within timeout."""
        return self.completed.wait(timeout=timeout)

    def get_status(self) -> dict:
        """Get current training status"""
        return {
            "status": self.current_status,
            "progress": self.current_progress,
            "success": self.success,
            "error": self.error_message,
            "completed": self.completed.is_set()
        }
