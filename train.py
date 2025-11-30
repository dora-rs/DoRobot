#!/usr/bin/env python3
"""
Simplified training client using only HTTP APIs - no database or backend dependencies.
This script uploads local training data directly to the server using SSH credentials
obtained from the API, monitors training progress, and downloads the trained model.
"""

import requests
import argparse
import base64
import os
import subprocess
import sys
import time
import json
from pathlib import Path

# Configuration (can be overridden by CLI args)
API_BASE_URL = "http://127.0.0.1:8000"
USERNAME = "userb"  # Change this to your username
PASSWORD = "userb1234"  # Change this to your password
DEFAULT_LOCAL_DATA_PATH = "/Users/nupylot/Public/so101-test-1127ok"  # Default local data folder path
DEFAULT_LOCAL_MODEL_OUTPUT = "/Users/nupylot/Public/so101-test-1127out"  # Default local model download path

def log(message):
    """Print timestamped log messages"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def login(api_url, username, password):
    """Login and get authentication token"""
    log("🔐 Logging in...")
    try:
        response = requests.post(
            f"{api_url}/login",
            json={"username": username, "password": password},
            timeout=60  # Increased for slow networks
        )
        response.raise_for_status()
        login_data = response.json()
        log(f"✅ Logged in as: {login_data['username']} (admin: {login_data['is_admin']})")
        return login_data["access_token"]
    except Exception as e:
        log(f"❌ Login failed: {e}")
        sys.exit(1)

def refresh_token_if_needed(api_url, username, password, current_token, force_refresh=False):
    """Refresh token if expired or force_refresh is True"""
    if force_refresh:
        log("🔄 Refreshing authentication token...")
        return login(api_url, username, password)
    return current_token

def start_transaction(api_url, token, training_config=None):
    """Start a new training transaction and get SSH credentials"""
    log("🚀 Starting transaction...")

    # Default training configuration
    if training_config is None:
        training_config = {
            "model_type": "act",
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "use_docker": False
        }

    try:
        response = requests.post(
            f"{api_url}/transactions/start",
            json={"training_config": training_config},
            headers={"Authorization": f"Bearer {token}"},
            timeout=60  # Increased for slow networks
        )
        response.raise_for_status()
        transaction_data = response.json()

        log(f"✅ Transaction started: {transaction_data['transaction_id']}")
        log(f"📁 S3 data path: {transaction_data['s3_data_path']}")
        log(f"🖥️  Instance ID: {transaction_data.get('instance_id', 'None')}")

        return transaction_data
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log(f"❌ Authentication failed when starting transaction - token may be expired")
            return None
        else:
            log(f"❌ Failed to start transaction: {e}")
            sys.exit(1)
    except Exception as e:
        log(f"❌ Failed to start transaction: {e}")
        sys.exit(1)

def setup_ssh_credentials(ssh_data):
    """Setup SSH credentials and return connection info"""
    log("🔑 Setting up SSH credentials...")

    # Decode base64 password
    ssh_password = base64.b64decode(ssh_data["ssh_password"]).decode('utf-8')

    ssh_info = {
        'host': ssh_data["ssh_host"],
        'username': ssh_data["ssh_username"],
        'password': ssh_password,
        'port': ssh_data["ssh_port"]
    }

    log(f"🌐 SSH Host: {ssh_info['username']}@{ssh_info['host']}:{ssh_info['port']}")
    return ssh_info

def test_ssh_connection(ssh_info):
    """Test SSH connection to ensure it works"""
    log("🔧 Testing SSH connection...")

    try:
        # Test basic SSH connection with options for slow networks
        cmd = [
            "sshpass", "-p", ssh_info['password'],
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=60",  # SSH connection timeout for slow networks
            "-o", "ServerAliveInterval=30",  # Keep connection alive
            "-o", "ServerAliveCountMax=10",  # Retry 10 times before disconnect
            "-p", str(ssh_info['port']),
            f"{ssh_info['username']}@{ssh_info['host']}",
            "echo 'SSH connection successful'"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # 2 minutes for slow networks

        if result.returncode == 0:
            log("✅ SSH connection test successful")
            return True
        else:
            log(f"❌ SSH connection test failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        log("❌ SSH connection test timed out")
        return False
    except Exception as e:
        log(f"❌ SSH connection test error: {e}")
        return False

def create_remote_directory(ssh_info, remote_path):
    """Create the remote directory structure"""
    log(f"📁 Creating remote directory: {remote_path}")

    try:
        cmd = [
            "sshpass", "-p", ssh_info['password'],
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-p", str(ssh_info['port']),
            f"{ssh_info['username']}@{ssh_info['host']}",
            f"mkdir -p {remote_path}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)  # 1.5 minutes for slow networks

        if result.returncode == 0:
            log(f"✅ Remote directory created: {remote_path}")
            return True
        else:
            log(f"❌ Failed to create remote directory: {result.stderr}")
            return False

    except Exception as e:
        log(f"❌ Error creating remote directory: {e}")
        return False

def rsync_data(ssh_info, local_path, remote_path):
    """Upload data using rsync over SSH"""
    log(f"📤 Starting rsync upload...")
    log(f"   Source: {local_path}")
    log(f"   Target: {ssh_info['username']}@{ssh_info['host']}:{remote_path}")

    try:
        # Rsync command with SSH and progress, optimized for slow networks
        ssh_options = (
            "sshpass -p '{password}' ssh "
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null "
            "-o LogLevel=ERROR "
            "-o ConnectTimeout=60 "
            "-o ServerAliveInterval=30 "
            "-o ServerAliveCountMax=20 "
            "-o TCPKeepAlive=yes "
            "-p {port}"
        ).format(password=ssh_info['password'], port=ssh_info['port'])

        cmd = [
            "rsync",
            "-avz",  # archive, verbose, compress
            "--progress",  # show progress
            "--delete",  # delete files in destination that don't exist in source
            "--timeout=1200",  # 20 minute timeout for individual operations
            "-e", ssh_options,
            f"{local_path}/",  # trailing slash to sync contents, not the folder itself
            f"{ssh_info['username']}@{ssh_info['host']}:{remote_path}/"
        ]

        log("🔄 Running rsync command...")

        # Run rsync with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in process.stdout:
            line = line.strip()
            if line:
                if "%" in line and "remaining" in line:
                    # Progress line
                    log(f"   📊 {line}")
                elif line.startswith("sending incremental") or line.startswith("sent ") or line.startswith("total size"):
                    # Summary lines
                    log(f"   ℹ️  {line}")
                elif not line.startswith("building file list") and not line.startswith("./"):
                    # Other important messages, skip file listings
                    if len(line) < 100:  # Avoid very long lines
                        log(f"   📄 {line}")

        process.wait()

        if process.returncode == 0:
            log("✅ Rsync upload completed successfully")
            return True
        else:
            log(f"❌ Rsync failed with return code: {process.returncode}")
            return False

    except Exception as e:
        log(f"❌ Rsync error: {e}")
        return False

def mark_upload_complete(api_url, token, transaction_id, force=False):
    """Mark the upload as complete to trigger training"""
    if force:
        log("✅ Marking upload as complete (FORCE mode)...")
    else:
        log("✅ Marking upload as complete...")

    try:
        # Add force parameter if enabled
        params = {"force": "true"} if force else {}
        response = requests.post(
            f"{api_url}/transactions/{transaction_id}/upload-complete",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=60  # Increased for slow networks
        )
        response.raise_for_status()
        result = response.json()

        log(f"✅ Upload marked complete: {result.get('message', 'Success')}")
        log(f"🎯 Training status: {result.get('status', 'Unknown')}")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log(f"❌ Authentication failed - token may be expired")
            return False
        else:
            try:
                error_detail = e.response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            log(f"❌ Failed to mark upload complete: {error_detail}")
            return False

    except Exception as e:
        log(f"❌ Failed to mark upload complete: {e}")
        return False

def check_training_status(api_url, token, transaction_id):
    """Check the current training status via API"""
    try:
        response = requests.get(
            f"{api_url}/transactions/{transaction_id}/status",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60  # Increased for slow networks
        )

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return {"status": "AUTH_ERROR", "error": "Token expired"}
        else:
            return {"status": "ERROR", "error": f"HTTP {response.status_code}"}

    except Exception as e:
        return {"status": "ERROR", "error": str(e)}

def monitor_training_via_ssh(ssh_info, transaction_id, check_interval=30, timeout_minutes=180):
    """Monitor training completion via SSH - simplified version without database"""
    log("🔍 Starting SSH-based training monitoring...")
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        try:
            # Check if training process is still running
            cmd = [
                "sshpass", "-p", ssh_info['password'],
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-p", str(ssh_info['port']),
                f"{ssh_info['username']}@{ssh_info['host']}",
                f"ps -ef | grep {transaction_id} | grep -v grep"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            active_processes = result.stdout.strip()

            if not active_processes:
                log(f"🔍 No active training processes found for {transaction_id}")

                # Check for model files in common locations
                model_locations = [
                    f"/root/admin-data/outputs/train/act_{transaction_id}/checkpoints/last/pretrained_model",
                    f"/root/autodl-tmp/outputs/train/act_{transaction_id}/checkpoints/last/pretrained_model",
                    f"/root/lerobot/scripts/outputs/train/act_{transaction_id}/checkpoints/last/pretrained_model",
                    f"/root/output/{transaction_id}/checkpoints/last/pretrained_model",
                    # Fallback: check parent directories
                    f"/root/admin-data/outputs/train/act_{transaction_id}",
                    f"/root/autodl-tmp/outputs/train/act_{transaction_id}",
                    f"/root/lerobot/scripts/outputs/train/act_{transaction_id}",
                    f"/root/output/{transaction_id}"
                ]

                for model_path in model_locations:
                    cmd = [
                        "sshpass", "-p", ssh_info['password'],
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null",
                        "-o", "LogLevel=ERROR",
                        "-p", str(ssh_info['port']),
                        f"{ssh_info['username']}@{ssh_info['host']}",
                        f"test -d '{model_path}' && echo 'EXISTS'"
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                    if result.stdout.strip() == "EXISTS":
                        log(f"✅ Found model directory: {model_path}")
                        return {"completed": True, "model_path": model_path}

                log("⚠️  No model files found yet, training may still be initializing...")
            else:
                log(f"⏳ Training process still active for {transaction_id}")

            time.sleep(check_interval)

        except Exception as e:
            log(f"❌ SSH monitoring error: {e}")
            time.sleep(check_interval)

    log(f"⏰ SSH monitoring timeout after {timeout_minutes} minutes")
    return {"completed": False, "error": "Timeout"}

def rsync_download_folder(ssh_info, remote_folder_path, local_output_path):
    """Download a remote folder to a local path using rsync over SSH"""
    log("📥 Starting rsync download of model folder...")
    log(f"   Remote: {ssh_info['username']}@{ssh_info['host']}:{remote_folder_path}")
    log(f"   Local:  {local_output_path}")

    try:
        # Ensure local output directory exists
        Path(local_output_path).mkdir(parents=True, exist_ok=True)

        # Trailing slash to copy contents of the folder
        remote_src = f"{remote_folder_path.rstrip('/')}/"

        # SSH options for slow networks (same as upload)
        ssh_options = (
            "sshpass -p '{password}' ssh "
            "-o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null "
            "-o LogLevel=ERROR "
            "-o ConnectTimeout=60 "
            "-o ServerAliveInterval=30 "
            "-o ServerAliveCountMax=20 "
            "-o TCPKeepAlive=yes "
            "-p {port}"
        ).format(password=ssh_info['password'], port=ssh_info['port'])

        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "--timeout=600",  # 10 minute timeout for individual operations
            "-e", ssh_options,
            f"{ssh_info['username']}@{ssh_info['host']}:{remote_src}",
            f"{local_output_path.rstrip('/')}/"
        ]

        log("🔄 Running rsync download command...")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            line = line.strip()
            if line:
                if "%" in line and "remaining" in line:
                    log(f"   📊 {line}")
                elif line.startswith("sent ") or line.startswith("total size"):
                    log(f"   ℹ️  {line}")

        process.wait()
        if process.returncode == 0:
            log("✅ Rsync download completed successfully")
            return True
        else:
            log(f"❌ Rsync download failed with return code: {process.returncode}")
            return False
    except Exception as e:
        log(f"❌ Rsync download error: {e}")
        return False

def modify_config_device(local_output_path, from_device="npu", to_device="cuda"):
    """Modify config.json file to change device after download"""
    config_path = Path(local_output_path) / "config.json"

    if not config_path.exists():
        log(f"⚠️  config.json not found at {config_path}")
        return False

    try:
        log(f"🔧 Modifying config.json device from '{from_device}' to '{to_device}'...")

        # Read the config file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Check if device field exists
        if 'device' not in config_data:
            log(f"⚠️  'device' field not found in config.json")
            return False

        original_device = config_data.get('device', 'unknown')
        log(f"   📄 Current device: {original_device}")

        # Modify the device field
        config_data['device'] = to_device

        # Write back to file
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)

        log(f"✅ Successfully updated device from '{original_device}' to '{to_device}' in config.json")
        return True

    except json.JSONDecodeError as e:
        log(f"❌ Failed to parse config.json: {e}")
        return False
    except Exception as e:
        log(f"❌ Failed to modify config.json: {e}")
        return False

def main():
    """Main workflow - simplified without database dependencies"""
    parser = argparse.ArgumentParser(description="Simple training client using only HTTP APIs")
    parser.add_argument("--api-url", default=API_BASE_URL,
                        help="API base URL (default: %(default)s)")
    parser.add_argument("--username", default=USERNAME,
                        help="Username for authentication (default: %(default)s)")
    parser.add_argument("--password", default=PASSWORD,
                        help="Password for authentication")
    parser.add_argument("--input", "--data", dest="input_path", default=DEFAULT_LOCAL_DATA_PATH,
                        help="Path to local dataset folder to upload (default: %(default)s)")
    parser.add_argument("--output", dest="output_path", default=DEFAULT_LOCAL_MODEL_OUTPUT,
                        help="Local folder to save the trained model (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: %(default)s)")
    parser.add_argument("--model-type", default="act",
                        help="Model type (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Training timeout in minutes (default: %(default)s)")
    parser.add_argument("--train-only", type=str, metavar="TRANSACTION_ID",
                        help="Skip upload and only start training for existing transaction ID")
    parser.add_argument("--force", action="store_true",
                        help="Force training even if transaction is completed/failed (use with --train-only)")
    args = parser.parse_args()

    local_data_path = args.input_path
    local_model_output = args.output_path

    log("🚀 Starting simplified training client")
    log(f"📡 API URL: {args.api_url}")

    # Store credentials for re-authentication if needed
    auth_credentials = {
        'api_url': args.api_url,
        'username': args.username,
        'password': args.password
    }

    # Step 1: Login
    token = login(args.api_url, args.username, args.password)

    # Check if we're in train-only mode
    if args.train_only:
        log(f"🎯 Train-only mode: Using existing transaction {args.train_only}")
        transaction_id = args.train_only

        # Skip to marking upload complete and monitoring
        upload_marked = mark_upload_complete(args.api_url, token, transaction_id, force=args.force)
        if not upload_marked:
            log("⚠️  First attempt to mark upload complete failed, trying to re-authenticate...")
            token = login(args.api_url, args.username, args.password)
            log("🔄 Re-authenticated with fresh token, retrying...")
            if not mark_upload_complete(args.api_url, token, transaction_id, force=args.force):
                log("❌ Failed to mark upload complete even after re-authentication")
                sys.exit(1)

        # Jump to monitoring section
        log("🔄 Monitoring training progress...")
        completion_detected = False
        model_path = None
        start_time = time.time()
        max_wait_time = args.timeout * 60  # Convert to seconds

    else:
        # Normal workflow with upload
        # Check if local data path exists
        if not os.path.exists(local_data_path):
            log(f"❌ Local data path does not exist: {local_data_path}")
            sys.exit(1)

        log(f"📁 Local data path: {local_data_path}")

        # Step 2: Start transaction with training config
        training_config = {
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_docker": False
        }

        transaction_data = start_transaction(args.api_url, token, training_config)

        # Retry with fresh token if authentication failed
        if transaction_data is None:
            log("⚠️  Authentication failed, refreshing token and retrying...")
            token = login(args.api_url, args.username, args.password)
            transaction_data = start_transaction(args.api_url, token, training_config)
            if transaction_data is None:
                log("❌ Failed to start transaction even after re-authentication")
                sys.exit(1)

        transaction_id = transaction_data["transaction_id"]

        # Check if SSH credentials are available
        if not all(k in transaction_data for k in ["ssh_host", "ssh_username", "ssh_password", "ssh_port"]):
            log("❌ SSH credentials not available in transaction response")
            log("   This might mean no instance was allocated or SSH is not configured")
            sys.exit(1)

        # Step 3: Setup SSH credentials
        ssh_info = setup_ssh_credentials(transaction_data)

        # Step 4: Test SSH connection
        if not test_ssh_connection(ssh_info):
            log("❌ SSH connection test failed, cannot proceed with rsync")
            sys.exit(1)

        # Step 5: Calculate remote path (add /root/ prefix)
        s3_data_path = transaction_data["s3_data_path"]
        remote_path = f"/root/{s3_data_path}"
        log(f"🎯 Remote target path: {remote_path}")

        # Step 6: Create remote directory
        if not create_remote_directory(ssh_info, remote_path):
            log("❌ Failed to create remote directory")
            sys.exit(1)

        # Step 7: Upload data using rsync
        if not rsync_data(ssh_info, local_data_path, remote_path):
            log("❌ Data upload failed")
            sys.exit(1)

        # Step 8: Mark upload complete (with retry on auth failure)
        upload_marked = mark_upload_complete(args.api_url, token, transaction_id, force=args.force)
        if not upload_marked:
            log("⚠️  First attempt to mark upload complete failed, trying to re-authenticate...")
            # Try to re-login and retry
            token = login(args.api_url, args.username, args.password)
            log("🔄 Re-authenticated with fresh token, retrying...")
            if not mark_upload_complete(args.api_url, token, transaction_id, force=args.force):
                log("❌ Failed to mark upload complete even after re-authentication")
                sys.exit(1)

        # Step 9: Monitor training progress
        log("🔄 Monitoring training progress...")

        completion_detected = False
        model_path = None
        start_time = time.time()
        max_wait_time = args.timeout * 60  # Convert to seconds

    # Unified monitoring loop for both modes
    while not completion_detected and (time.time() - start_time) < max_wait_time:
        # Check via SSH only if we have SSH credentials (normal mode)
        if not args.train_only and 'ssh_info' in locals():
            ssh_result = monitor_training_via_ssh(ssh_info, transaction_id,
                                                 check_interval=30,
                                                 timeout_minutes=1)  # Quick check

            if ssh_result.get("completed"):
                log("🎉 Training completion detected via SSH!")
                model_path = ssh_result.get("model_path")
                completion_detected = True
                break

        # Check API status (available in both modes)
        api_status = check_training_status(args.api_url, token, transaction_id)
        current_status = api_status.get("status", "UNKNOWN")
        completed_at = api_status.get("completed_at")

        log(f"   📊 API Status: {current_status}")

        # Check for completion - either status is COMPLETED or completed_at is set
        if current_status == "COMPLETED" or completed_at:
            if completed_at:
                log("🎉 Training completion detected via completed_at timestamp!")
                log(f"   ✅ Completed at: {completed_at}")
            else:
                log("🎉 Training completion detected via API status!")
            model_path = api_status.get("model_path")
            completion_detected = True
            break
        elif current_status in ("FAILED", "ERROR", "CANCELLED"):
            err = api_status.get("error_message", "")
            log(f"❌ Training failed with status: {current_status}")
            if err:
                log(f"   Error: {err}")
            sys.exit(1)
        elif current_status == "AUTH_ERROR":
            log("⚠️  Authentication error - token may be expired")
            # Try to re-login
            token = login(args.api_url, args.username, args.password)
            log("🔄 Re-authenticated with fresh token")

        time.sleep(10)  # Check every 10 seconds

    # Check final results
    if not completion_detected:
        log(f"⏰ Maximum wait time ({args.timeout} minutes) exceeded - training may still be in progress")
        sys.exit(1)

    if not model_path:
        log("❌ Training completed but no model_path available for download")
        sys.exit(1)

    # Step 10: Download the trained model
    log(f"✅ Training completed. Model path: {model_path}")
    log(f"📥 Downloading model to local: {local_model_output}")

    if args.train_only:
        # Train-only mode: Use API download endpoint
        log("🔗 Train-only mode: Using API download endpoint...")
        try:
            response = requests.get(
                f"{args.api_url}/transactions/{transaction_id}/download-model",
                headers={"Authorization": f"Bearer {token}"},
                timeout=60
            )
            response.raise_for_status()
            download_data = response.json()
            log(f"📦 Download URL: {download_data.get('download_url', 'Available via API')}")
            log("ℹ️  Please use the web interface or API to download the model in train-only mode")
        except Exception as e:
            log(f"⚠️  API download info failed: {e}")
            log("ℹ️  Model is available for download via the web interface")
    else:
        # Normal mode: Use SSH/rsync download
        if not rsync_download_folder(ssh_info, model_path, local_model_output):
            log("❌ Failed to download model folder via rsync")
            sys.exit(1)

        # Step 11: Post-processing - modify config.json
        log("🔧 Post-download processing: updating config.json device setting...")
        if not modify_config_device(local_model_output, from_device="npu", to_device="cuda"):
            log("⚠️  Warning: Failed to update config.json device setting, but continuing...")

    log("🎉 Training workflow completed successfully!")
    log(f"   Transaction ID: {transaction_id}")
    if not args.train_only:
        log(f"   Data uploaded to: {remote_path}")
        log(f"   Model downloaded to: {local_model_output}")
    else:
        log("   Training started for existing transaction")
    log("\n📝 Usage examples:")
    if args.train_only:
        log(f"   Train-only mode: python train.py --train-only {transaction_id}")
    else:
        log(f"   Full workflow: python train.py --input {local_data_path} --output {local_model_output}")
        log(f"   Train-only mode: python train.py --train-only TRANSACTION_ID")

if __name__ == "__main__":
    main()
