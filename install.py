import launch

dependencies = [
    "ultralytics",
    "opencv-python",
    "numpy",
    "Pillow"
]

for d in dependencies:
    if not launch.is_installed(d):
        launch.run_pip(f"install {d}", d)
