# this script will perform the complete SfM reconstruction, Gaussian Splatting training, and launch the visualizer
import subprocess


GAUSSIAN_SPLAT_PATH = "C:/github-repos/gaussian-splatting/"
SFM_EXE_NAME = "sfm_ceres.exe"
PYTHON = "python"
CONDA = "conda"
CONDA_ENV_NAME = "gaussian_splatting"


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

def conda():
    try:
        result = subprocess.run([CONDA, "activate", CONDA_ENV_NAME], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the conda process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


def train_gaussian_splat():
    gaussian_splat = GAUSSIAN_SPLAT_PATH + "train.py" # TODO: correct path + check if absolute is okay
    flag_source = "--source_path=../out/window_with_anchor/" # TODO: correct path + check if relative is okay
    flag_model = "--model_path=C:\\github-repos\\sfm-gaussian-splatting\\sfm_cpp\\out\\trained"
    flag_iters = "--iterations=7000"
    flag_debug = "--debug"
    # flags = "-s C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/ -m C:/github-repos/sfm-gaussian-splatting/sfm_cpp/out/trained/ --iterations 7000" # TODO: correct path + check if relative is okay
    try:
        #result = subprocess.run([PYTHON, gaussian_splat, flag_source, flag_model, flag_iters, flag_debug], check=True, capture_output=True, text=True)
        result = subprocess.run([PYTHON, gaussian_splat, flag_source, flag_model, flag_iters], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the Gaussian Splatting training process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


def render():
    render = GAUSSIAN_SPLAT_PATH + "render.py" # TODO: correct path + check if absolute is okay
    flags = "-m ../out/trained"
    try:
        result = subprocess.run([PYTHON, render, flags], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the rendering process.")
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)


if __name__ == "__main__":
    #sfm()
    #conda()
    train_gaussian_splat()
    #render()