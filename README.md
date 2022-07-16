# Camera Calibration using OpenCV

There exist many resources for Camera Calibration and this has become a standard operating procedure. Here, we use OpenCV to calibrate single or stereo cameras using chessboard or ChArUco boards. We emphasize the importance of high-quality calibration boards that are on flat surfaces and not warped.

Alternative calibration options include importing calibration data produced with the [Calibrator calib.io Software](https://calib.io/products/calib), which relies on a different matrix-optimization approach and other flexible parameters. A python-based conversion code is available that converts output from Calibrator to the OpenCV XML camera matrix format.

**Table of Contents**
- [Preparation](#preparation)
    - [Configuring the camera](#configuring-the-camera)
    - [Calibration targets](#calibration-targets)
    - [Preparing the scene](#preparing-the-scene)
- [Usage](#usage)
    - [Examples](#examples)


## Preparation
Before capturing calibration images it is critical to ensure the camera is configured  and mounted properly, the board is high quality, and the scene is set appropriately to ensure a smooth calibration process.

### Configuring the camera

#### Exposure settings
Calibration works best when the camera's internal settings (e.g. f-stop, shutter speed,  ISO) are fixed and constant for the calibration session. The cameras used here were placed into "Manual" mode (a feature on the majority of high-quality digital cameras today) and f-stop, shutter speed, and ISO manually set according to the lighting conditions at the time of the sessions.

#### Minimizing blur
When taking the photos it is important to minimize blur, so if it is not possible to use a remote shutter when capturing the images, consider enabling a short self-timer to avoid vibration at the time of capture.

The Sony cameras used in this study were also equipped with a "SteadyShot" feature which was disabled prior to shooting because, when enabled, it results in small adjustments being made to the internal mechanics of the lens to physically eliminate motion blur, but at the cost of slightly altering the lens's internal parameters and resulting in poorer calibration results. It is recommended that any automatic blur-reduction features are disabled.

The camera should be mounted on a tripod to ensure that its internal and geometric parameters remain constant throughout the session. **This is crucial for a good calibration and handheld photos result in a significantly worse result.**

### Calibration targets<a name="calibration-targets" />
We recommend the use of a high-quality calibration board such as the aluminum composite checkerboard or CharuCo targets made by [Calib.io](https://calib.io/), because any inconsistencies in the target's flatness or pattern will affect the calibration results. Target stiffness is also important as it is recommended to place the board in multiple orientations (often requiring it to be leaned at an angle against another object).

We found that the checkerboard targets tended to provid better calibration results, but because the entire checkerboard needs to be within the frame at all times it can be somewhat tricky to align it perfectly in the corners (where the most distortion often is found). The CharuCo targets help with this issue as they do not need to be entirely in the frame, but at the cost of *slightly* worse results.

<table><tr>
    <td> 
      <p align="center" style="padding: 10px; text-align: center;">
        <img alt="Checkerboard" src="img/checker.jpg" width="300">
        <br>
        <em style="color: grey">Calib.io Checkerboard Target</em>
      </p> 
    </td>
    <td> 
      <p align="center" style="padding: 10px; text-align: center;">
        <img alt="CharuCo" src="img/charuco.jpg" width="300">
        <br>
        <em style="color: grey">Calib.io CharuCo Target</em>
      </p> 
    </td>
</tr></table>

### Preparing the scene<a name="preparing-the-scene" />
<table><tr>
    <td> 
      <p align="center" style="padding: 10px; text-align: center;">
        <img alt="Our camera setup" src="img/setup.jpg" width="300">
        <br>
        <em style="color: grey">A good setup makes things simple.</em>
      </p> 
    </td>
</tr></table>

If performing the calibration indoors, a well-lit environment with minimal shadows is key. LED lighting panels are good for this. Place the tripod with the camera in front of the target area and tilt the camera to a near-orthogonal orientation to the ground. For quick and consistent target placement during the shooting, we found it helped to first place tape along the edge of the camera frame. This can be done by placing the target in one corner of the frame as precisely as possible and then using it as a guide to place the tape. Repeat for the other three corners.

Once you have the frame marked out, we recommend taking 20-30 photos with the target covering every part of the frame at least once. Placing the target at oblique angles to the camera is also recommended, and for this we simply leaned the target up against an item in the lab and attempted to capture it in several different areas of the frame.

<table><tr>
    <td> 
      <p align="center" style="padding: 10px; text-align: center;">
        <img alt="Target placed at an oblique angle" src="img/oblique.jpg" width="300">
        <br>
        <em style="color: grey">A checkerboard target placed at an oblique angle</em>
      </p> 
    </td>
</tr></table>

## Usage<a name="usage" />
The parameters have been optimized for the cameras that are available at the Geological Remote Sensing lab at the University of Potsdam. These are several Sony alpha-6 (24 MP), Sony alpha-7 (40 MP), and Fuji X-100 (24 MP) all with 55 mm or 85 mm fixed lenses.

The python code `python/single_camera_calibration_charuco_chess_openCV.py` can be called from the command line. Use `single_camera_calibration_charuco_chess_openCV.py -h` to obtain a short help and description of the parameters.

The code will read all calibration images from one directory, plots a summary figure showing all photos, performs the camera calibration, and writes the distortion coefficient and intrinsic camera calibration to an OpenCV XML file.

**Example of 49 photos showing a chessboard pattern for camera calibration (Sony alpha-7 55 mm lense):**
![](examples/camA_chessboard_49images.jpg)

**All detected chessboard intersection - make sure that points are also taken from the corners of the image (Sony alpha-7 55 mm lense):**
![](examples/camA_chess_23324corners.png)

**Example camera calibration and pixel distortion using intrinsic and distorted parameters (Sony alpha-7 55 mm lense):**
![](examples/sony_alpha7_55m_CC_chess_comparison_1panel.jpg)

## Examples
Example call from Ubuntu command line (expecting OpenCV to be installed).

#### Using Sony alpha-6000 and charuco board

```bash
camA_initial_CC='cam_A_calib_9parameters_fine_charuco_20Feb2022.xml'
charuco_ifiles_camA='sony_stereo_f13_iso1600/charuco/black_a_stereo/DSC*.JPG'
camA_charuco_savexml_file='sony_stereo_f13_iso1600/charuco/cam_A_black_calib_9parameters_fine_charuco_25Mar2022.xml'
camA_CC_comparison_3panel_png='sony_stereo_f13_iso1600/charuco/CC_comparison_3panel.png'
camA_CC_comparison_1panel_png='sony_stereo_f13_iso1600/charuco/CC_comparison_1panel.png'
camA_Height=4000
camA_Width=6000
focal_length_pixels=9000

single_camera_calibration_charuco_chess_openCV.py --camA_initial_CC $camA_initial_CC \
  --charuco_ifiles_camA $charuco_ifiles_camA \
  --camA_charuco_savexml_file $camA_charuco_savexml_file \
  --camA_CC_comparison_3panel_png $camA_CC_comparison_3panel_png \
  --camA_CC_comparison_1panel_png $camA_CC_comparison_1panel_png \
  --camA_Height $camA_Height --camA_Width $camA_Width \
  --focal_length_pixels $focal_length_pixels
```

#### Using Sony alpha-7000 with fixed 55 mm lense and chess board

Using no initial calibration file and this requires setting the parameters *camA_Height*, *camA_Width*, and
*focal_length_pixels*.

```bash
chess_ifiles_camA='near/DSC*.JPG'
camA_chess_savexml_file='sony_alpha7_55m_CC_05July2022.xml'
camA_chess_75pbest_savexml_file='sony_alpha7_55m_CC_75p_05July2022.xml'
camA_CC_comparison_3panel_png='sony_alpha7_55m_CC_chess_comparison_3panel.png'
camA_CC_comparison_1panel_png='sony_alpha7_55m_CC_chess_comparison_1panel.png'
camA_Height=5304
camA_Width=7952
focal_length_pixels=12675

single_camera_calibration_charuco_chess_openCV.py  \
  --chess_ifiles_camA $chess_ifiles_camA \
  --camA_chess_savexml_file $camA_chess_savexml_file \
  --camA_chess_75pbest_savexml_file $camA_chess_75pbest_savexml_file \
  --camA_CC_comparison_3panel_png $camA_CC_comparison_3panel_png \
  --camA_CC_comparison_1panel_png $camA_CC_comparison_1panel_png \
  --camA_Height $camA_Height --camA_Width $camA_Width \
  --focal_length_pixels $focal_length_pixels
```

#### Using Sony alpha-7000 with fixed 85 mm lense and chess board

Using no initial calibration file and this requires setting parameters for camera calibration.

```bash
chess_ifiles_camA='near/DSC*.JPG'
camA_chess_savexml_file='sony_alpha7_85m_CC_05July2022.xml'
camA_chess_75pbest_savexml_file='sony_alpha7_85m_CC_75p_05July2022.xml'
camA_CC_comparison_3panel_png='sony_alpha7_85m_CC_chess_comparison_3panel.png'
camA_CC_comparison_1panel_png='sony_alpha7_85m_CC_chess_comparison_1panel.png'
camA_Height=5304
camA_Width=7952
focal_length_pixels=18918

single_camera_calibration_charuco_chess_openCV.py  \
  --chess_ifiles_camA $chess_ifiles_camA \
  --camA_chess_savexml_file $camA_chess_savexml_file \
  --camA_chess_75pbest_savexml_file $camA_chess_75pbest_savexml_file \
  --camA_CC_comparison_3panel_png $camA_CC_comparison_3panel_png \
  --camA_CC_comparison_1panel_png $camA_CC_comparison_1panel_png \
  --camA_Height $camA_Height --camA_Width $camA_Width \
  --focal_length_pixels $focal_length_pixels
```
