import subprocess
import os
import sys

GIT_EXE = r"C:\Users\paruc\AppData\Local\GitHubDesktop\app-3.6.5\resources\app\git\cmd\git.exe"

# Clean up temp files
temp_files = [
    "check_large_files.py",
    "check_models_size.py",
    "find_git.py",
    "find_git_deep.py",
    "run_git.py",
    "test_git_run.py"
]
for f in temp_files:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Removed temp file: {f}")
        except Exception as e:
            print(f"Failed to remove {f}: {e}")

def run_git(args, check=True):
    cmd = [GIT_EXE] + args
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print("STDERR:")
        print(res.stderr.strip())
    print(f"EXIT CODE: {res.returncode}")
    if check and res.returncode != 0:
        print(f"Command failed with code {res.returncode}")
    return res

# 1. Check current remotes
run_git(["remote", "-v"], check=False)

# 2. Check current status
run_git(["status"], check=False)

# 3. Add all changes
run_git(["add", "-A"])

# 4. Commit changes if any
commit_res = run_git(["commit", "-m", "TrustGuard AI full functional platform with trained models and end-to-end UI"], check=False)

# 5. Check if remote 'reshma' exists or update 'origin'
remotes_res = run_git(["remote"], check=False)
remotes = remotes_res.stdout.split()

TARGET_URL = "https://github.com/reshma630025/reshma.git"

if "reshma" in remotes:
    run_git(["remote", "set-url", "reshma", TARGET_URL])
else:
    run_git(["remote", "add", "reshma", TARGET_URL])

# Also update origin to point to target
run_git(["remote", "set-url", "origin", TARGET_URL])

# 6. Verify remotes
run_git(["remote", "-v"])

# 7. Check current branch
branch_res = run_git(["branch", "--show-current"])
current_branch = branch_res.stdout.strip() or "main"
print(f"Current branch is: {current_branch}")

# 8. Force push to TARGET_URL
print(f"\n>>> Executing force push to {TARGET_URL} on branch {current_branch}...")
push_res = run_git(["push", "--force", TARGET_URL, f"{current_branch}:main"], check=False)
if push_res.returncode == 0:
    print("\nSUCCESS! Successfully force-pushed to " + TARGET_URL)
else:
    print(f"\nPush returned exit code {push_res.returncode}")
