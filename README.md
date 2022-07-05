# Camera Calibration using OpenCV

There exist many resources for Camera Calibration and this has become a standard operating procedure. Here, we use OpenCV to calibrate single or stereo cameras using chessboard or ChArUco boards. We emphasize the importance of high-quality calibration boards that are on flat surfaces and not warped.

Alternative calibration options include importing calibration data produced with the [Calibrator calib.io Software](https://calib.io/products/calib), which relies on a different matrix-optimization approach and other flexible parameters. A python-based conversion code is available that converts output from Calibrator to the OpenCV XML camera matrix format.

## Usage
The parameters have been optimized for the cameras that are available at the Geological Remote Sensing lab at the University of Potsdam. These are several Sony alpha-6 (24 MP), Sony alpha-7 (40 MP), and Fuji X-100 (24 MP) all with 55 mm or 85 mm fixed lenses.

The python code `python/single_camera_calibration_charuco_chess_openCV.py` can be called from the command line. Use `single_camera_calibration_charuco_chess_openCV.py -h` to obtain a short help and description of the parameters.

The code will read all calibration images from one directory, plots a summary figure showing all photos, performs the camera calibration, and writes the distortion coefficient and intrinsic camera calibration to an OpenCV XML file.

**Example of 49 photos showing a chessboard pattern for camera calibration (Sony alpha-7 55 mm lense):**
![](examples/camA_chessboard_49images.jpg)

**All detected chessboard intersection - make sure that points are also taken from the corners of the image (Sony alpha-7 55 mm lense):**
![](examples/camA_chess_23324corners.png)

**Example camera calibration and pixel distortion using intrinsic and distorted parameters (Sony alpha-7 55 mm lense):**
![](examples/sony_alpha7_55m_CC_chess_comparison_1panel.jpg)

### Examples
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
  --camA_Height $camA_Height --camA_Width $camA_Width --focal_length_pixels $focal_length_pixels
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
  --camA_Height $camA_Height --camA_Width $camA_Width --focal_length_pixels $focal_length_pixels
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
  --camA_Height $camA_Height --camA_Width $camA_Width --focal_length_pixels $focal_length_pixels
```
