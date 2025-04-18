# sfm-gaussian-splatting
A Github repository for Bob and John's CMPUT 428 (Computer Vision) final project.

We created an incremental Structure-from-Motion (SfM) algorithm similar to COLMAP [3,4] to reconstruct 3D scenes as sparse 3D points clouds. Our algorithm uses elementary techniques taught in CMPUT 428 and Hartley and Zisserman [5].

We used Gaussian Splatting [1,2] to visualize create a rendering of our results, and compared the rendering quality associated with accuracy of the the 3D point clouds used for initialization.

<p align="center">
  <img src="https://github.com/john00003/sfm-gaussian-splatting/blob/john_reverted_matching/sfm_cpp/out/colmap_minimal_more/COLMAP%20data%20demo.gif" width=600 height=400/>
  <img src="https://github.com/john00003/sfm-gaussian-splatting/blob/john_reverted_matching/sfm_cpp/out/colmap_minimal_more/splat_smaller.gif" width=600 height=400/>
</p>

## Installation Instructions

### SfM Installation
Check the releases tab for a downloadable version. This release packages all of the dependencies the project was built with into .dll files. This version should work out of the box on any Window's machine equipped with a Nvidia graphics card.

### Gaussian Splatting Training + Viewer Installation
The Gaussian Splatting repository [2] has complete instructions for installation of the Gaussian Splatting training program, and a visualization tool to view the rendered scene. Follow the installation tutorial (complete with a video tutorial for Windows) in the repository's README.

This will involve creating a Conda environment for Python dependencies. Although the Gaussian Splatting repository recommends CUDA v11.8, we used CUDA v12.4, which is compatible with C++ compiler included in the latest version of Visual Studio Community 2022. This made building the CUDA related dependencies much easier.

## Running Instructions

### Running the SfM Algorithm
Navigate to `sfm_cpp/out/build/x64-Release` and run `./sfm_ceres.exe`, input the relative path to the images of the 3D scene you would like to reconstruct into the GUI.

Once the program has finished, check `sfm_cpp/out/` for three output text files (not required for Gaussian Splatting, but human readable), and a folder titled `sparse/`.

### Running the Gaussian Splatting Training
First, create a folder that contains the following:
* a folder title `images/` containing the images used to run the SfM algorithm
* the folder containing the binary output by the SfM algorithm, `sparse/`

Then, navigate to the directory where you installed the Gaussian Splatting repository. Use the command `python train.py -s <path to folder you created> -m <path to folder you created (or where you'd like the output to be)> --iterations=7000`, specifying the number of iterations you would like.

### Running the Gaussian Splatting Viewer

Navigate to the directory where you installed the Gaussian Splatting viewer. Use the command `./bin/SIBR_gaussianViewer_app -m <path to trained model (the folder specified by -m in previous command)>`.




## References

[1] Gaussian Splatting (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

[2] Gaussian Splatting Github Repository (https://github.com/graphdeco-inria/gaussian-splatting)

[3] COLMAP (https://colmap.github.io/)

[4] Structure-from-Motion Revisited (https://openaccess.thecvf.com/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)

[5] Multiple View Geometry in Computer Vision (https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A)

---

<div align="right">

<img src="UofAlbertalogo.svg" alt="University of Alberta Logo" width="330px" style="vertical-align: middle;">
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
Department of Computing Science
</p>
<p style="margin: 0; font-size: 14px; font-weight: bold;">
April 2025, Edmonton, AB, Canada
</p>

</div>
