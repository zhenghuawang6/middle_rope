import subprocess

def get_git_commit_hash():
    try:
        # 执行 git rev-parse HEAD 命令
        result = subprocess.run(
            ["git", "rev-parse", "--short" , "HEAD"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            check=True
        )
        return result.stdout.strip()  # 移除多余的换行符
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.strip()}")
        return None