python /home/bodo/Dropbox/soft/github/CameraCalibration/python 
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/calib-to-opencv.py --json_dir ./ --xml_dir ./xml/

```bash
#35 mm: 2 vs. 5 parameters
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_free_2p_25Apr2025.xml --title0 'Sony 7RM5 - 35 mm - free checkerboard - 2p' \
  --CC1_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_free_5p_25Apr2025.xml --title1 'Sony 7RM5 - 35 mm - free checkerboard - 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_35mm_checkerboard_free_2p5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_35mm_checkerboard_free_2p5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 35mm free checkerboard - 2 vs 5 param.' --h 6336 --w 9504

#35 mm: 5 parameters targetoptimization+robustfit
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_free_5p_targetoptimization_robustfit_25Apr2025.xml --title0 'Sony 7RM5 - 50 mm - free checkerboard - 5p (rb, to)' \
  --CC1_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_free_5p_25Apr2025.xml --title1 'Sony 7RM5 - 35 mm - free checkerboard - 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_35mm_checkerboard_free_5p_robustfit_targetoptimization_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_35mm_checkerboard_free_5p_robustfit_targetoptimization_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 35mm free checkerboard - 5 param. targetopt. + robustfit' --h 6336 --w 9504

#35 mm: 5 parameters fixed vs. free
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_free_5p_25Apr2025.xml --title0 'free checkerboard 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_fixed_5p_25Apr2025.xml --title1 'fixed checkerboard 5p' \
  --save_fname_2panel Sony_ILCE-7RM5_35mm_checkerboard_free_fixed_5p_25Apr2025_2panel.png \
  --save_fname_2panel_diff Sony_ILCE-7RM5_35mm_checkerboard_free_fixed_5p_25Apr2025_2panel_diff.png \
  --save_fname_1panel Sony_ILCE-7RM5_35mm_checkerboard_free_5p_25Apr2025_1panel_diff.png \
  --title_diff 'Offset Difference fixed and free checkerboard 5 p' \
  --suptitle 'Sony 7RM5 35 mm'
rsync Sony_ILCE-7RM5_35mm_checkerboard_free_fixed_5p_25Apr2025_2panel.png Sony_ILCE-7RM5_35mm_checkerboard_free_fixed_5p_25Apr2025_2panel_diff.png \
/home/bodo/Dropbox/soft/github/up-rs-esp.github.io/_posts/CameraCalibration1_images


#35 mm: 5 parameters fixed robustfit
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_fixed_5p_25Apr2025.xml --title0 'Sony 7RM5 - 35 mm - fixed checkerboard - 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_fixed_5p_robustnorm_25Apr2025.xml --title1 'Sony 7RM5 - 35 mm - fixed checkerboard w/ robustfit - 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_35mm_checkerboard_fixed_5p_robustnorm_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_35mm_checkerboard_fixed_5p_robustnorm_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 35 mm fixed checkerb. robustfit - 5 p' --h 6336 --w 9504
`
# 50 mm: 5 parameters fixed vs. free
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_free_5p_25Apr2025.xml --title0 'free checkerboard 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_fixed_5p_25Apr2025.xml --title1 'fixed checkerboard 5p' \
  --save_fname_2panel Sony_ILCE-7RM5_50mm_checkerboard_free_fixed_5p_25Apr2025_2panel.png \
  --save_fname_2panel_diff Sony_ILCE-7RM5_50mm_checkerboard_free_fixed_5p_25Apr2025_2panel_diff.png \
  --save_fname_1panel Sony_ILCE-7RM5_50mm_checkerboard_free_fixed_5p_25Apr2025_1panel.png \
   --title_diff 'Offset Difference fixed and free checkerboard 5 p' \
  --suptitle 'Sony 7RM5 50 mm'

rsync Sony_ILCE-7RM5_50mm_checkerboard_free_fixed_5p_25Apr2025_2panel.png Sony_ILCE-7RM5_50mm_checkerboard_free_fixed_5p_25Apr2025_2panel_diff.png \
/home/bodo/Dropbox/soft/github/up-rs-esp.github.io/_posts/CameraCalibration1_images

# 50 mm: 5 parameters fixed checkerboard vs. charuco - THIS IS BAD
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_fixed_5p_25Apr2025.xml --title0 'Sony 7RM5 50 mm - fixed checkerboard 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_50mm_charuco_fixed_5p_robustnorm_25Apr2025.xml --title1 'Sony 7RM5 50 mm - fixed charuco robustfit 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_50mm_checkerboard_charuco_fixed_5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_50mm_checkerboard_charuco_fixed_5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 50 mm fixed checkerboard and charuco 5'

# 50 mm: 5 parameters fixed checkerboard vs. free charuco
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_fixed_5p_25Apr2025.xml --title0 'Sony 7RM5 50 mm - fixed checkerboard 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_50mm_charuco_free_5p_25Apr2025.xml --title1 'Sony 7RM5 50 mm - free charuco 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_50mm_checkerboard_charuco_fixed_5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_50mm_checkerboard_charuco_fixed_5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 50 mm fixed checkerboard and free charuco 5'


# 50 mm: 5 parameters charuco fixed vs. free
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_charuco_fixed_5p_robustnorm_25Apr2025.xml --title0 'Sony 7RM5 50 mm - fixed charuco robustfit 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_50mm_charuco_free_5p_25Apr2025.xml --title1 'Sony 7RM5 50 mm - free charuco 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_50mm_charuco_fixed_free_5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_50mm_charuco_fixed_free_5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 50 mm fixed vs. free charuco 5p' 


# 35 mm: 5 parameters fixed checkerboard vs. charuco
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_35mm_checkerboard_free_5p_25Apr2025.xml --title0 'Sony 7RM5 35 mm - free checkerboard 5p' \
  --CC1_fn xml/Sony_ILCE-7RM5_35mm_charuco_free_5p_robustnorm_25Apr2025.xml --title1 'Sony 7RM5 35 mm - free charuco robustfit 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_35mm_checkerboard_charuco_free_5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_35mm_checkerboard_charuco_free_5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 35 mm free checkerboard and charuco 5p'



# 50 mm: 5 parameters fixed vs. Metashape optimization
python /home/bodo/Dropbox/soft/github/CameraCalibration/python/compareDistortion_from_CC_xml.py \
  --CC0_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_fixed_5p_25Apr2025_adjusted_outside_1_123images_202ktiepoints.xml --title0 'Sony 7RM5 50 mm - fix. checkerb. 5p - Metashape' \
  --CC1_fn xml/Sony_ILCE-7RM5_50mm_checkerboard_fixed_5p_25Apr2025.xml --title1 'Sony 7RM5 50 mm - fix. checkerb. - 5p' \
  --save_fname_3panel Sony_ILCE-7RM5_50mm_checkerboard_fixed_MetashapeOptimization_5p_25Apr2025_3panel.png \
  --save_fname_1panel Sony_ILCE-7RM5_50mm_checkerboard_fixed_MetashapeOptimization_5p_25Apr2025_1panel.png \
  --title_diff 'Sony 7RM5 50 mm fixed and free checkerb. - 5 p' --h 6336 --w 9504

```
