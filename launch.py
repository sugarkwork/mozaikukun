import subprocess
import os
import sys

python = sys.executable
skip_install = False


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
    if desc is not None:
        print(desc)
    if live:
        result = subprocess.run(command, shell=True, env=os.environ if custom_env is None else custom_env)
        if result.returncode != 0:
            raise RuntimeError(f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}""")
        return ""
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ if custom_env is None else custom_env)

    if result.returncode != 0:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {result.returncode}
stdout: {result.stdout.decode(encoding="utf8", errors="ignore") if len(result.stdout)>0 else '<empty>'}
stderr: {result.stderr.decode(encoding="utf8", errors="ignore") if len(result.stderr)>0 else '<empty>'}
"""
        raise RuntimeError(message)
    
    return result.stdout.decode(encoding="utf8", errors="ignore")


def run_pip(args, desc=None):
    if skip_install:
        return
    
    return run(f'"{python}" -m pip {args} --prefer-binary', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}")


def prepare_environment():
    global skip_install

    requirements_file = os.environ.get('REQS_FILE', "requirements.txt")
    sys.argv, skip_install = extract_arg(sys.argv, '--skip-install')
    run_pip(f"install -r {requirements_file}", "requirements for Web UI")


def extract_arg(args, name):
    return [x for x in args if x != name], name in args


if __name__ == "__main__":
    prepare_environment()

    run(f'"{python}" webui.py', "Starting Web UI", "Couldn't start Web UI", live=True)
