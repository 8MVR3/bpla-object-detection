import subprocess


def get_git_commit_id():
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"
