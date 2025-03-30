# this script will perform the complete SfM reconstruction, Gaussian Splatting training, and launch the visualizer
import subprocess


GAUSSIAN_SPLAT_PATH = "C:\github-repos\gaussian-splatting"
SFM_EXE_NAME = "sfm_ceres.exe"


def sfm():
    sfm = "../out/" + SFM_EXE_NAME  # TODO: correct path + check if relative is okay
    flags = "--no_gui" # TODO: update c++ to check flag
    try:
        result = subprocess.run([sfm, flags], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the SfM process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


def train_gaussian_splat():
    gaussian_splat = GAUSSIAN_SPLAT_PATH + "\train.py" # TODO: correct path + check if absolute is okay
    flags = "-s ../out/ -m ../out/trained" # TODO: correct path + check if relative is okay
    try:
        result = subprocess.run([gaussian_splat, flags], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the Gaussian Splatting training process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


def render():
    render = GAUSSIAN_SPLAT_PATH + "\render.py" # TODO: correct path + check if absolute is okay
    flags = "-m ../out/trained"
    try:
        result = subprocess.run([render, flags], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the rendering process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


if __name__ == "__main__":
    sfm()
    train_gaussian_splat()
    render()