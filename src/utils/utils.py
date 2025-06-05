import subprocess


def get_git_commit_id():
    try:
        commit_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return commit_id
    except Exception:
        return "unknown"
