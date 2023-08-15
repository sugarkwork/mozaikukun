import launch
import os
import pkg_resources

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

def dist2package(dist: str):
    return ({
        "opencv-python": "cv2",
        "Pillow": "PIL"
    }).get(dist, dist)

with open(req_file, encoding='UTF-8') as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != package_version:
                    launch.run_pip(f"install {package}", f"mozaikunkun requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(dist2package(package)):
                launch.run_pip(f"install {package}", f"mozaikunkun requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, something may not work.')
