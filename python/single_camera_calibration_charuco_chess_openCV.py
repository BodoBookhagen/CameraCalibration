# /usr/bin/env python
""" Single Camera Calibration using either charuco or chess boards (bodo.bookhagen@uni-potsdam.de)

"""

import numpy as np
import os, glob, datetime, tqdm, argparse
import cv2
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.path import Path

# SUB PIXEL CORNER DETECTION CRITERION
subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
CC_criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 1000, 1e-9)
stereo_calib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)

matrices = {
    "true": [
        [0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.299, 0.587, 0.114],
    ],
    "mono": [
        [0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114],
    ],
    "color": [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
    "halfcolor": [[0.299, 0.587, 0.114, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
    "optimized": [[0, 0.7, 0.3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 1]],
}


def plot_img_corners(cam_corners, png_fname, title, w=6000, h=4000, fontsize=10):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    for imagei in range(len(cam_corners)):
        ax.plot(
            np.squeeze(np.asarray(cam_corners[imagei]))[:, 0],
            np.squeeze(np.asarray(cam_corners[imagei]))[:, 1],
            "+",
            color="dodgerblue",
        )
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    fig.tight_layout()
    fig.savefig(png_fname, dpi=300)
    plt.close()


def visualizeDistortion(
    save_fname_1panel,
    K0,
    D0,
    h,
    w,
    fontsize=10,
    contourLevels=10,
    nstep=100,
    title0="cam0",
):

    D = D0.ravel()
    d = np.zeros(14)
    d[: D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u, v = np.meshgrid(np.arange(0, w, nstep), np.arange(0, h, nstep))
    b = np.array([u.ravel(), v.ravel(), np.ones(u.size)])
    xyz = np.linalg.lstsq(K0, b, rcond=None)[0]
    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + D[5] * r2 + D[6] * r4 + D[7] * r6)
    xpp = (
        xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    )
    ypp = (
        yp * coef
        + p1 * (r2 + 2 * yp**2)
        + 2 * p2 * (xp * yp)
        + D[11] * r2
        + D[11] * r4
    )
    u2 = K0[0, 0] * xpp + K0[0, 2]
    v2 = K0[1, 1] * ypp + K0[1, 2]
    du0 = np.copy(u2.ravel() - u.ravel())
    dv0 = np.copy(v2.ravel() - v.ravel())
    dr0 = np.copy(np.hypot(du0, dv0).reshape(u.shape))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=600)
    ax.quiver(u.ravel(), v.ravel(), du0, -dv0, color="dodgerblue")
    ax.plot(w // 2, h // 2, "x", K0[0, 2], K0[1, 2], "^", markersize=fontsize)
    CS = ax.contour(u, v, dr0, colors="black", levels=contourLevels)
    ax.set_aspect("equal", "box")
    ax.clabel(CS, inline=1, fontsize=fontsize, fmt="%0.0f")
    ax.set_title(title0, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_ylim(max(v.ravel()), 0)
    fig.tight_layout()
    fig.savefig(save_fname_1panel, dpi=300)
    plt.close()


def visualizeDistortion_comparison_2(
    save_fname_3panel,
    save_fname_1panel,
    K0,
    D0,
    K1,
    D1,
    h,
    w,
    fontsize=16,
    contourLevels=10,
    nstep=20,
    title0="cam0",
    title1="cam1",
    title_diff="Difference",
):
    D = D0.ravel()
    d = np.zeros(14)
    d[: D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u, v = np.meshgrid(np.arange(0, w, nstep), np.arange(0, h, nstep))
    b = np.array([u.ravel(), v.ravel(), np.ones(u.size)])
    xyz = np.linalg.lstsq(K0, b, rcond=None)[0]
    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + D[5] * r2 + D[6] * r4 + D[7] * r6)
    xpp = (
        xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    )
    ypp = (
        yp * coef
        + p1 * (r2 + 2 * yp**2)
        + 2 * p2 * (xp * yp)
        + D[11] * r2
        + D[11] * r4
    )
    u2 = K0[0, 0] * xpp + K0[0, 2]
    v2 = K0[1, 1] * ypp + K0[1, 2]
    du0 = np.copy(u2.ravel() - u.ravel())
    dv0 = np.copy(v2.ravel() - v.ravel())
    dr0 = np.copy(np.hypot(du0, dv0).reshape(u.shape))

    fig, ax = plt.subplots(1, 3, figsize=(12, 8), dpi=600)
    ax[0].quiver(u.ravel(), v.ravel(), du0, -dv0, color="dodgerblue")
    ax[0].plot(w // 2, h // 2, "x", K0[0, 2], K0[1, 2], "^", markersize=fontsize)
    CS = ax[0].contour(u, v, dr0, colors="black", levels=contourLevels)
    ax[0].set_aspect("equal", "box")
    ax[0].clabel(CS, inline=1, fontsize=fontsize, fmt="%0.0f")
    ax[0].set_title(title0, fontsize=fontsize)
    ax[0].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax[0].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax[0].tick_params(axis="both", which="major", labelsize=fontsize)
    ax[0].set_ylim(max(v.ravel()), 0)

    D = D1.ravel()
    d = np.zeros(14)
    d[: D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u, v = np.meshgrid(np.arange(0, w, nstep), np.arange(0, h, nstep))
    b = np.array([u.ravel(), v.ravel(), np.ones(u.size)])
    xyz = np.linalg.lstsq(K1, b, rcond=None)[0]
    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + D[5] * r2 + D[6] * r4 + D[7] * r6)
    xpp = (
        xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    )
    ypp = (
        yp * coef
        + p1 * (r2 + 2 * yp**2)
        + 2 * p2 * (xp * yp)
        + D[11] * r2
        + D[11] * r4
    )
    u2 = K1[0, 0] * xpp + K1[0, 2]
    v2 = K1[1, 1] * ypp + K1[1, 2]
    du1 = np.copy(u2.ravel() - u.ravel())
    dv1 = np.copy(v2.ravel() - v.ravel())
    dr1 = np.copy(np.hypot(du1, dv1).reshape(u.shape))

    ax[1].quiver(u.ravel(), v.ravel(), du1, -dv1, color="dodgerblue")
    ax[1].plot(w // 2, h // 2, "x", K1[0, 2], K1[1, 2], "^", markersize=fontsize)
    CS = ax[1].contour(u, v, dr1, colors="black", levels=contourLevels)
    ax[1].set_aspect("equal", "box")
    ax[1].clabel(CS, inline=1, fontsize=fontsize, fmt="%0.0f")
    ax[1].set_title(title1, fontsize=fontsize)
    ax[1].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax[1].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
    ax[1].set_ylim(max(v.ravel()), 0)

    ax[2].quiver(u.ravel(), v.ravel(), (du1 - du0), (-dv1 + dv0), color="dodgerblue")
    ax[2].plot(
        w // 2,
        h // 2,
        "x",
        K0[0, 2],
        K0[1, 2],
        "o",
        K1[0, 2],
        K1[1, 2],
        "+",
        markersize=fontsize,
    )
    CS = ax[2].contour(u, v, dr1 - dr0, colors="black", levels=contourLevels)
    ax[2].set_aspect("equal", "box")
    ax[2].clabel(CS, inline=1, fontsize=fontsize, fmt="%0.0f")
    ax[2].set_title(title_diff, fontsize=fontsize)
    ax[2].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax[2].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax[2].tick_params(axis="both", which="major", labelsize=fontsize)
    ax[2].set_ylim(max(v.ravel()), 0)
    fig.tight_layout()
    fig.savefig(save_fname_3panel, dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    ax.quiver(u.ravel(), v.ravel(), (du1 - du0), (-dv1 + dv0), color="dodgerblue")
    ax.plot(
        w // 2,
        h // 2,
        "x",
        K0[0, 2],
        K0[1, 2],
        "o",
        K1[0, 2],
        K1[1, 2],
        "+",
        markersize=fontsize,
    )
    CS = ax.contour(u, v, dr1 - dr0, colors="black", levels=contourLevels)
    ax.set_aspect("equal", "box")
    ax.clabel(CS, inline=1, fontsize=fontsize, fmt="%0.0f")
    ax.set_title(title_diff, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_ylim(max(v.ravel()), 0)
    fig.tight_layout()
    fig.savefig(save_fname_1panel, dpi=300)
    plt.close()


def save_opencv_CC_xml(savexml_file, image_size, cameraMatrix, distortion_coefficient):
    print("Writing CC to xml: %s" % savexml_file)
    cv_file = cv2.FileStorage(savexml_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("calibration_Time", str(datetime.datetime.now()))
    cv_file.write("image_Height", image_size[0])
    cv_file.write("image_Width", image_size[1])
    cv_file.write("Camera_Matrix", cameraMatrix)
    cv_file.write("Distortion_Coefficients", distortion_coefficient)
    cv_file.release()


def load_stereo_coefficients(path):
    """Loads stereo matrix coefficients."""
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


def save_stereo_coefficients(path, K0, D0, K1, D1, R, T, E, F, R0, R1, P0, P1, Q):
    """Save the stereo coefficients to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K1", K0)
    cv_file.write("D1", D0)
    cv_file.write("K2", K1)
    cv_file.write("D2", D1)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.write("E", E)
    cv_file.write("F", F)
    cv_file.write("R1", R0)
    cv_file.write("R2", R1)
    cv_file.write("P1", P0)
    cv_file.write("P2", P1)
    cv_file.write("Q", Q)
    cv_file.release()


def save_fundamental_essential_matrix(path, F, E):
    """Save the Fundamental and Essential Matrix to given path/file."""
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("F", F)
    cv_file.write("E", E)
    cv_file.release()


def load_CC_from_XML(fname):
    cv_file = cv2.FileStorage(fname, cv2.FILE_STORAGE_READ)
    cv_file.open(fname, cv2.FILE_STORAGE_READ)
    H = cv_file.getNode("image_Height").real()
    W = cv_file.getNode("image_Width").real()
    K = cv_file.getNode("Camera_Matrix").mat()
    D = cv_file.getNode("Distortion_Coefficients").mat()
    cv_file.release()
    return K, D, H, W


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def computeReprojectionErrors(imgpoints, objpoints, rvecs, tvecs, K, D):
    """
    Uses the camera matrix (K) and the distortion coefficients to reproject the
    object points back into 3D camera space and then calculate the error between
    them and the image points that were found.
    Reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    imgpoints: features found in image, (num_imgs, 2)
    objpoints: calibration known features in 3d, (num_imgs, 3)
    rvecs: rotations
    tvecs: translations
    K: camera matrix
    D: distortion coefficients [k1,k2,p1,p2,k3]
    returns:
        rms
        rms_per_view
        errors
    """
    imgpoints = [c.reshape(-1, 2) for c in imgpoints]
    mean_error = None
    error_x = []
    error_y = []
    rms = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = imgpoints2.reshape(-1, 2)

        # if not all markers were found, then the norm below will fail
        if len(imgpoints[i]) != len(imgpoints2):
            continue

        error_x.append(list(imgpoints2[:, 0] - imgpoints[i][:, 0]))
        error_y.append(list(imgpoints2[:, 1] - imgpoints[i][:, 1]))

        rr = np.sum((imgpoints2 - imgpoints[i]) ** 2, axis=1)
        if mean_error is None:
            mean_error = rr
        else:
            mean_error = np.hstack((mean_error, rr))
        rr = np.sqrt(np.mean(rr))
        rms.append(rr)

    m_error = np.sqrt(np.mean(mean_error))
    return m_error, rms, [error_x, error_y]


def visualizeReprojErrors(
    totalRSME, rmsPerView, reprojErrs, fontSize=12, legend=False, xlim=None, ylim=None
):
    fig, ax = plt.subplots()
    for i, (rms, x, y) in enumerate(zip(rmsPerView, reprojErrs[0], reprojErrs[1])):
        ax.scatter(x, y, label=f"RMSE [{i}]: {rms:0.3f}")

    # change dimensions if legend displayed
    if legend:
        ax.legend(fontsize=fontSize)
        ax.axis("equal")
    else:
        ax.set_aspect("equal", "box")
    ax.set_title(
        f"Reprojection Error (in pixels), RMSE: {totalRSME:0.4f}", fontsize=fontSize
    )
    ax.set_xlabel("X", fontsize=fontSize)
    ax.set_ylabel("Y", fontsize=fontSize)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="major", labelsize=fontSize)
    ax.grid(True)
    plt.show()


def find_charuco_markers(
    image_fn,
    cameraMatrix,
    distCoeffs,
    CHARUCO_BOARD,
    CHARUCO_DICT=cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000),
    criteria=subpixel_criteria,
):
    """
    Find Charuca Markers in list of images
    """
    allCorners = []
    allIds = []
    decimator = 0

    # see http://www.bim-times.com/opencv/4.3.0/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
    # These will need to be adjusted if images are of lower quality
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 64
    parameters.adaptiveThreshWinSizeStep = 2
    parameters.adaptiveThreshConstant = 27

    image_true_fn = []
    for fname in image_fn:
        print("reading %s, " % os.path.basename(fname), flush=True, end="")
        frame = cv2.imread(
            fname, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray,
            dictionary=CHARUCO_DICT,
            parameters=parameters,
            cameraMatrix=cameraMatrix,
            distCoeff=distCoeffs,
        )

        # # Draw and display the corners
        # fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
        # plt1=ax.imshow(gray, cmap='gray')
        # for i in range(len(corners)):
        #     ax.plot(corners[i].squeeze()[:,0], corners[i].squeeze()[:,1],'rx', ms=5, mew=0.2, ls="none")
        # for i in range(len(rejectedImgPoints)):
        #     ax.plot(rejectedImgPoints[i].squeeze()[:,0], rejectedImgPoints[i].squeeze()[:,1],'bo', ms=5, mew=0.2, ls="none")

        print("found %d corners, " % len(corners), flush=True, end="")
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(
                    gray,
                    corner,
                    winSize=(3, 3),
                    zeroZone=(-1, -1),
                    criteria=subpixel_criteria,
                )
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, CHARUCO_BOARD
            )
            if (
                res2[1] is not None
                and res2[2] is not None
                and len(res2[1]) > 3
                and decimator % 1 == 0
            ):
                print("sub-pixel detection and storing to array.")
                allCorners.append(res2[1])
                allIds.append(res2[2])
                image_true_fn.append(fname)
            else:
                print("no sub-pixels found.")

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize, image_true_fn


def calibrate_camera_charuco(
    allCorners,
    allIds,
    imsize,
    cameraMatrixInit,
    distCoeffsInit,
    calib_flags,
    CHARUCO_BOARD,
    CC_criteria,
):
    """
    Calibrates the camera using the dected corners.
    """

    # distCoeffsInit = np.zeros((5,1))
    (
        ret,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
        stdDeviationsIntrinsics,
        stdDeviationsExtrinsics,
        perViewErrors,
    ) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=CHARUCO_BOARD,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=calib_flags,
        criteria=CC_criteria,
    )

    return (
        ret,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    )


def CC_charucoboard(
    image_files,
    cameraMatrixInit,
    distCoeffsInit,
    w,
    h,
    savepng_output=True,
    savepng_corners_output_file="",
    savepng_path="",
    CHARUCOBOARD_ROWCOUNT=18,
    CHARUCOBOARD_COLCOUNT=25,
    square_size_m=0.015,
    marker_length_m=0.012,
    CHARUCO_DICT=cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000),
):
    # # Creating vector to store vectors of 2D points for each charucoboard image
    # imgpoints = []

    # Creating vector to store vectors of 3D points for each charucoboard image
    # objpoints = []
    # # Defining the world coordinates for 3D points - only for chessboard
    # objp = np.zeros((1, CHARUCOBOARD_ROWCOUNT * CHARUCOBOARD_COLCOUNT, 3), np.float32)
    # objp[0,:,:2] = np.mgrid[0:CHARUCOBOARD_ROWCOUNT, 0:CHARUCOBOARD_COLCOUNT].T.reshape(-1, 2)
    # objp = objp * square_size_m  # Create real world coords. Use your metric.
    # prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    if type(image_files) == list:
        image_fn = image_files
        image_directory = os.path.dirname(image_files[0])
    else:
        image_fn = glob.glob(image_files)
        image_directory = os.path.dirname(image_files)
    image_fn.sort()

    if len(image_fn) == 0:
        print("no image files found in %s" % image_fn)
        return

    print("Loading images from directory %s..." % image_directory)
    # CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)
    # Create constants to be passed into OpenCV and Aruco methods
    # CHARUCOBOARD_ROWCOUNT = 18
    # CHARUCOBOARD_COLCOUNT = 25
    # square_size_m = 0.015
    # marker_length_m=0.012
    # CHARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=square_size_m,
        markerLength=marker_length_m,
        dictionary=CHARUCO_DICT,
    )

    allCorners, allIds, imsize, image_true_fn = find_charuco_markers(
        image_fn,
        cameraMatrixInit,
        distCoeffsInit,
        CHARUCO_BOARD,
        CHARUCO_DICT=CHARUCO_DICT,
        criteria=subpixel_criteria,
    )
    # allCorners = allCorners * square_size_m

    if savepng_output == True:
        if len(savepng_corners_output_file) == 0:
            savepng_corners_output_file = "charucoboard_%02dimages.png" % len(
                image_true_fn
            )
        if len(savepng_corners_output_file) > 0:
            savepng_corners_output_file = (
                savepng_corners_output_file + "%02dimages.png" % len(image_true_fn)
            )
        savepng_corners_output_file = os.path.join(
            savepng_path, savepng_corners_output_file
        )

        nr_images = len(allCorners)
        nr_rows = np.int8(np.round(np.sqrt(nr_images)))
        nr_cols = np.int8(np.ceil(np.sqrt(nr_images)))

        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(12, 8), dpi=300)
        plotx, ploty = 0, 0
        print("Plotting image i of %02d: " % nr_images, end="", flush=True)
        for icorner in range(len(image_true_fn)):
            print("%02d, " % (icorner + 1), end="", flush=True)
            img = cv2.imread(
                image_true_fn[icorner],
                flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = allCorners[icorner]

            # Draw and display the corners
            plt1 = ax[plotx, ploty].imshow(gray, cmap="gray")
            ax[plotx, ploty].set_aspect("equal", "box")
            for i in range(corners.shape[0]):
                ax[plotx, ploty].plot(
                    corners[i].squeeze()[0],
                    corners[i].squeeze()[1],
                    "rx",
                    ms=2,
                    mew=0.2,
                    ls="none",
                )
                ax[plotx, ploty].set_title(
                    "%s (corners: %02d)"
                    % (os.path.basename(image_true_fn[icorner]), len(corners)),
                    fontsize=8,
                )
                # ax[plotx, ploty].axis('tight')
                ax[plotx, ploty].axis("off")
            # ax[plotx, ploty].tick_params(
            #     bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
            if ploty == nr_cols - 1:
                ploty = 0
                plotx = plotx + 1
            else:
                ploty = ploty + 1
        fig.tight_layout()
        fig.savefig(savepng_corners_output_file, dpi=300)
        plt.close()
        print("done.")

    calib_flags = 0
    calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # calib_flags |= cv2.CALIB_FIX_INTRINSIC
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # calib_flags |= cv2.CALIB_RATIONAL_MODEL
    # calib_flags |= cv2.CALIB_FIX_K1
    # calib_flags |= cv2.CALIB_FIX_K2
    # calib_flags |= cv2.CALIB_FIX_K3
    # calib_flags |= cv2.CALIB_FIX_K4
    # calib_flags |= cv2.CALIB_FIX_K5
    # calib_flags |= cv2.CALIB_FIX_K6
    (
        RMS,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
    ) = calibrate_camera_charuco(
        allCorners,
        allIds,
        imsize,
        cameraMatrixInit.copy(),
        distCoeffsInit.copy(),
        calib_flags,
        CHARUCO_BOARD,
        CC_criteria,
    )
    allCorners_sum = sum([len(listElem) for listElem in allCorners])
    # print('Nr. of corners: %05d'%allCorners_sum)

    print("")
    print("Difference to initial calibration file from charuco calibration")
    print("rms : %2.2f" % RMS)
    print(
        "cx: %+02.2f \t Delta cx from initial calibration: %2.4f"
        % (camera_matrix[0][2], cameraMatrixInit[0][2] - camera_matrix[0][2])
    )
    print(
        "cy: %+02.2f \t Delta cy from initial calibration: %2.4f"
        % (camera_matrix[1][2], cameraMatrixInit[1][2] - camera_matrix[1][2])
    )
    print(
        "k1: %+02.4f \t Delta k1 from initial calibration: %2.4f"
        % (distortion_coefficients[0], distCoeffsInit[0] - distortion_coefficients[0])
    )
    print(
        "k2: %+02.4f \t Delta k2 from initial calibration: %2.4f"
        % (distortion_coefficients[1], distCoeffsInit[1] - distortion_coefficients[1])
    )
    print(
        "k3: %+02.4f \t Delta k3 from initial calibration: %2.4f"
        % (distortion_coefficients[4], distCoeffsInit[4] - distortion_coefficients[4])
    )
    print(
        "p1: %+02.4f \t Delta p1 from initial calibration: %2.4f"
        % (distortion_coefficients[2], distCoeffsInit[2] - distortion_coefficients[2])
    )
    print(
        "p2: %+02.4f \t Delta p2 from initial calibration: %2.4f"
        % (distortion_coefficients[3], distCoeffsInit[3] - distortion_coefficients[3])
    )
    print("")
    print("Nr. of images: \t\t\t\t %05d" % len(allCorners))
    print("Nr. of corners points: \t %05d" % allCorners_sum)
    print("Original Camera Matrix: \n", cameraMatrixInit)
    print("New Camera Matrix: \n", camera_matrix)
    print("")
    print("Original Distortion Coefficients: \n", distCoeffsInit)
    print("New Distortion Coefficients: \n", distortion_coefficients)
    return (
        allCorners,
        allIds,
        imsize,
        RMS,
        camera_matrix,
        distortion_coefficients,
        rotation_vectors,
        translation_vectors,
        CHARUCO_BOARD,
    )


def find_chessboard_markers(
    image_files,
    chessboard_width,
    chessboard_height,
    square_size_m=0.010,
    H=4000,
    W=6000,
):
    criteria = CC_criteria
    CHESSBOARD = (chessboard_width, chessboard_height)

    # Creating vector to store vectors of 3D points for each chessboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each chessboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0 : CHESSBOARD[0], 0 : CHESSBOARD[1]].T.reshape(-1, 2)
    objp = objp * square_size_m  # scale to real world coordinates

    image_true_fn = []
    image_size = np.array([H, W])

    if type(image_files) == list or type(image_files) == str:
        if type(image_files) == list:
            image_fn = image_files
        else:
            image_fn = glob.glob(image_files)

        image_fn.sort()

        if len(image_fn) == 0:
            print("no image files found in %s" % image_fn)
            return
        for fname in image_fn:
            print("reading %s, " % os.path.basename(fname), flush=True, end="")
            img = cv2.imread(
                fname, flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR
            )
            # if img.shape[1] < img.shape[0]:
            #     img = np.rot90(img, k=-1)
            #     print('rotating image, ', end='', flush=True)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape
            # print(image_size)
            # Find the chess board corners
            # If desired number of corners are found in the image then ret = true
            # ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            # CHESSBOARD_SCALING_FACTOR=chessboard_scaling
            # height_resize, width_resize = int(height/CHESSBOARD_SCALING_FACTOR), int(width/CHESSBOARD_SCALING_FACTOR)
            # grayS = cv2.resize(gray, (width_resize, height_resize))                    # Resize image
            ret, corners_S = cv2.findChessboardCorners(
                gray,
                CHESSBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FILTER_QUADS,
            )
            if ret == True:
                # corners = corners_S * CHESSBOARD_SCALING_FACTOR
                corners = corners_S
            else:
                print("no corners found.")
            """
            If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
            """
            if ret == True:
                print("found Chessboard Corners, ", end="", flush=True)
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
                print("found SubPixel corner.", flush=True)
                image_true_fn.append(fname)

                imgpoints.append(corners2)

    elif type(image_files) != list:
        if image_files.shape[0] > 1000 and image_files.shape[1] > 1000:
            image_size = image_files.shape
            ret, corners_S = cv2.findChessboardCorners(
                image_files,
                CHESSBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FILTER_QUADS,
            )
            if ret == True:
                # corners = corners_S * CHESSBOARD_SCALING_FACTOR
                corners = corners_S
            else:
                print("no corners found.")
            """
            If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
            """
            if ret == True:
                print("found Chessboard Corners, ", end="", flush=True)
                objpoints.append(objp)
                # refining pixel coordinates for given 2d points.
                corners2 = cv2.cornerSubPix(
                    image_files, corners, (7, 7), (-1, -1), criteria
                )
                print("found SubPixel corner.", flush=True)

                imgpoints.append(corners2)

    return objpoints, imgpoints, image_true_fn, image_size


def CC_chessboard(
    image_files,
    K_initial,
    D_initial,
    W0,
    H0,
    chessboard_width,
    chessboard_height,
    savepng_output=True,
    savepng_corners_output_file="",
    savepng_path="",
    savexml_file="",
    savexml_file_best75p="",
):

    objpoints, imgpoints, image_true_fn, image_size = find_chessboard_markers(
        image_files,
        chessboard_width=chessboard_width,
        chessboard_height=chessboard_height,
        square_size_m=0.010,
        H=H0,
        W=W0,
    )

    if savepng_output == True:
        if len(savepng_corners_output_file) == 0:
            savepng_corners_output_file = "chessboard_%02dimages.png" % len(objpoints)
        if len(savepng_corners_output_file) > 0:
            savepng_corners_output_file = (
                savepng_corners_output_file + "%02dimages.png" % len(objpoints)
            )
        savepng_corners_output_file = os.path.join(
            savepng_path, savepng_corners_output_file
        )

        nr_images = len(objpoints)
        nr_rows = np.int8(np.round(np.sqrt(nr_images)))
        nr_cols = np.int8(np.ceil(np.sqrt(nr_images)))
        fig, ax = plt.subplots(nr_rows, nr_cols, figsize=(12, 8), dpi=300)
        plotx, ploty = 0, 0
        print("Plotting image i of %02d: " % nr_images, end="", flush=True)
        for icorner in range(nr_images):
            print("%02d, " % (icorner + 1), end="", flush=True)
            img = cv2.imread(
                image_true_fn[icorner],
                flags=cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR,
            )
            # if img.shape[1] < img.shape[0]:
            #     img = np.rot90(img,k=-1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = imgpoints[icorner]

            # Draw and display the corners
            plt1 = ax[plotx, ploty].imshow(gray, cmap="gray")
            ax[plotx, ploty].set_aspect("equal", "box")
            ax[plotx, ploty]
            for i in range(corners.shape[0]):
                ax[plotx, ploty].plot(
                    corners[i].squeeze()[0],
                    corners[i].squeeze()[1],
                    "rx",
                    ms=2,
                    mew=0.2,
                    ls="none",
                )
                ax[plotx, ploty].set_title(
                    "%s" % (os.path.basename(image_true_fn[icorner])), fontsize=8
                )
                # ax[plotx, ploty].axis('tight')
                ax[plotx, ploty].axis("off")
            # ax[plotx, ploty].tick_params(
            #     bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off
            if ploty == nr_cols - 1:
                ploty = 0
                plotx = plotx + 1
            else:
                ploty = ploty + 1
        fig.tight_layout()
        fig.savefig(savepng_corners_output_file, dpi=300)
        plt.close()
        print("done.")

    """
    Performing camera calibration by passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the detected corners (imgpoints)
    """
    print(
        "Calibrating Camera and saving to XML. Assuming same x-y focal lengths and using intrinsic CC ... ",
        flush=True,
    )

    calib_flags = 0
    calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # calib_flags |= cv2.CALIB_FIX_INTRINSIC
    calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # calib_flags |= cv2.CALIB_RATIONAL_MODEL
    # calib_flags |= cv2.CALIB_FIX_K1
    # calib_flags |= cv2.CALIB_FIX_K2
    # calib_flags |= cv2.CALIB_FIX_K3
    # calib_flags |= cv2.CALIB_FIX_K4
    # calib_flags |= cv2.CALIB_FIX_K5
    # calib_flags |= cv2.CALIB_FIX_K6

    # calibrateCameraROExtended(objectPoints, imagePoints, imageSize, iFixedPoint, cameraMatrix,
    # distCoeffs[, rvecs[, tvecs[, newObjPoints[, stdDeviationsIntrinsics[, stdDeviationsExtrinsics[, stdDeviationsObjPoints
    # [, perViewErrors[, flags[, criteria]]]]]]]]]) ->
    # retval, cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, stdDeviationsIntrinsics,
    # stdDeviationsExtrinsics, stdDeviationsObjPoints, perViewErrors
    # This function is an extension of #calibrateCamera with the method of releasing object which was
    # proposed in @cite strobl2011iccv. In many common cases with inaccurate, unmeasured, roughly planar
    # targets (calibration plates), this method can dramatically improve the precision of the estimated
    # camera parameters. Both the object-releasing method and standard method are supported by this
    # function. Use the parameter **iFixedPoint** for method selection. In the internal implementation,
    # calibrateCamera is a wrapper for this function.
    (
        rmsRO,
        intrinsic_matrixRO,
        distortion_coefficientRO,
        rvecsRO,
        tvecsRO,
        newObjPointsRO,
        stdDevIntrinsicsRO,
        stdDevExtrinsicsRO,
        stdDevObjPointsRO,
        perViewErrorsRO,
    ) = cv2.calibrateCameraROExtended(
        objpoints,
        imgpoints,
        [W0, H0],
        1,
        K_initial.copy(),
        D_initial.copy(),
        flags=calib_flags,
        criteria=CC_criteria,
    )
    distortion_coefficientRO = np.squeeze(distortion_coefficientRO)
    stdDevIntrinsicsRO = np.squeeze(stdDevIntrinsicsRO)
    print("Image Size: ", [W0, H0])
    print("Difference to initial calibration file from RO calibration")
    print("rms (RO): %2.2f" % rmsRO)
    print(
        "cx (RO): %+02.2f ±%02.2f (1-σ percent %02.1f %%) \t\t Delta cx from initial calibration: %2.4f"
        % (
            intrinsic_matrixRO[0][2],
            stdDevIntrinsicsRO[2],
            np.abs(stdDevIntrinsicsRO[2] / intrinsic_matrixRO[0][2] * 100),
            K_initial[0][2] - intrinsic_matrixRO[0][2],
        )
    )
    print(
        "cy (RO): %+02.2f ±%02.2f (1-σ percent %02.1f %%) \t\t Delta cy from initial calibration: %2.4f"
        % (
            intrinsic_matrixRO[1][2],
            stdDevIntrinsicsRO[3],
            np.abs(stdDevIntrinsicsRO[3] / intrinsic_matrixRO[1][2] * 100),
            K_initial[1][2] - intrinsic_matrixRO[1][2],
        )
    )
    print(
        "k1 (RO): %+02.4f ±%02.4f (1-σ percent %02.1f %%) \t Delta k1 from initial calibration: %2.4f"
        % (
            distortion_coefficientRO[0],
            stdDevIntrinsicsRO[4],
            np.abs(stdDevIntrinsicsRO[4] / distortion_coefficientRO[0] * 100),
            D_initial[0] - distortion_coefficientRO[0],
        )
    )
    print(
        "k2 (RO): %+02.4f ±%02.4f (1-σ percent %02.1f %%) \t Delta k2 from initial calibration: %2.4f"
        % (
            distortion_coefficientRO[1],
            stdDevIntrinsicsRO[5],
            np.abs(stdDevIntrinsicsRO[5] / distortion_coefficientRO[1] * 100),
            D_initial[1] - distortion_coefficientRO[1],
        )
    )
    print(
        "k3 (RO): %+02.4f ±%02.4f (1-σ percent %02.1f %%) \t Delta k3 from initial calibration: %2.4f"
        % (
            distortion_coefficientRO[4],
            stdDevIntrinsicsRO[8],
            np.abs(stdDevIntrinsicsRO[8] / distortion_coefficientRO[4] * 100),
            D_initial[4] - distortion_coefficientRO[4],
        )
    )
    print(
        "p1 (RO): %+02.4f ±%02.4f (1-σ percent %02.1f %%) \t\t Delta p1 from initial calibration: %2.4f"
        % (
            distortion_coefficientRO[2],
            stdDevIntrinsicsRO[6],
            np.abs(stdDevIntrinsicsRO[6] / distortion_coefficientRO[2] * 100),
            D_initial[2] - distortion_coefficientRO[2],
        )
    )
    print(
        "p2 (RO): %+02.4f ±%02.4f (1-σ percent %02.1f %%) \t Delta p2 from initial calibration: %2.4f"
        % (
            distortion_coefficientRO[3],
            stdDevIntrinsicsRO[7],
            np.abs(stdDevIntrinsicsRO[7] / distortion_coefficientRO[3] * 100),
            D_initial[3] - distortion_coefficientRO[3],
        )
    )
    print("")
    print("Nr. of images: \t\t\t %05d" % len(objpoints))
    print("Nr. of chessboard points: \t %05d" % len(np.squeeze(objpoints[0])))
    print("")
    print("Original Camera Matrix: \n", K_initial)
    print("New Camera Matrix: \n", intrinsic_matrixRO)
    print("")
    print("Original Distortion Coefficients: \t", np.squeeze(D_initial))
    print("New Distortion Coefficients: \t\t", distortion_coefficientRO)
    print("")
    print("Writing CC to xml: %s" % savexml_file)
    cv_file = cv2.FileStorage(savexml_file, cv2.FILE_STORAGE_WRITE)
    cv_file.write("calibration_Time", str(datetime.datetime.now()))
    cv_file.write("image_Height", image_size[0])
    cv_file.write("image_Width", image_size[1])
    cv_file.write("Camera_Matrix", intrinsic_matrixRO)
    cv_file.write("Distortion_Coefficients", distortion_coefficientRO)
    cv_file.release()
    print("done.")

    # Repeat camera calibration using only 75% of best images with lowest perViewErrorsRO
    perViewErrorsRO_sort_idx = np.argsort(np.squeeze(perViewErrorsRO), axis=0)
    (perViewErrorsRO_sort_idx75,) = np.where(
        np.squeeze(perViewErrorsRO)[perViewErrorsRO_sort_idx]
        < np.percentile(np.squeeze(perViewErrorsRO), 75)
    )
    objpoints_75 = np.array(objpoints)[perViewErrorsRO_sort_idx75]
    imgpoints_75 = np.array(imgpoints)[perViewErrorsRO_sort_idx75]

    (
        rmsRO_75,
        intrinsic_matrixRO_75,
        distortion_coefficientRO_75,
        rvecsRO_75,
        tvecsRO_75,
        newObjPointsRO_75,
        stdDevIntrinsicsRO_75,
        stdDevExtrinsicsRO_75,
        stdDevObjPointsRO_75,
        perViewErrorsRO_75,
    ) = cv2.calibrateCameraROExtended(
        objpoints_75,
        imgpoints_75,
        image_size,
        1,
        K_initial.copy(),
        D_initial.copy(),
        flags=calib_flags,
        criteria=CC_criteria,
    )
    distortion_coefficientRO_75 = np.squeeze(distortion_coefficientRO_75)
    stdDevIntrinsicsRO_75 = np.squeeze(stdDevIntrinsicsRO_75)

    print("")
    print(
        "Difference between first RO calibration and RO calibration with lowest 75p perViewErrorsRO"
    )
    print("First rms (RO): %2.2f \t\t New rms (RO): %2.2f" % (rmsRO, rmsRO_75))
    print(
        "cx (RO): %+2.2f ±%2.2f (1-σ percent %2.1f %%) \t\t Delta cx from initial calibration: %2.5f"
        % (
            intrinsic_matrixRO_75[0][2],
            stdDevIntrinsicsRO_75[2],
            np.abs(stdDevIntrinsicsRO_75[2] / intrinsic_matrixRO_75[0][2] * 100),
            intrinsic_matrixRO[0][2] - intrinsic_matrixRO_75[0][2],
        )
    )
    print(
        "cy (RO): %+2.2f ±%2.2f (1-σ percent %2.1f %%) \t\t Delta cy from initial calibration: %2.5f"
        % (
            intrinsic_matrixRO_75[1][2],
            stdDevIntrinsicsRO_75[3],
            np.abs(stdDevIntrinsicsRO_75[3] / intrinsic_matrixRO_75[1][2] * 100),
            intrinsic_matrixRO[1][2] - intrinsic_matrixRO_75[1][2],
        )
    )
    print(
        "k1 (RO): %+2.4f ±%2.4f (1-σ percent %2.1f %%) \t\t Delta k1 from initial calibration: %2.5f"
        % (
            distortion_coefficientRO_75[0],
            stdDevIntrinsicsRO_75[4],
            np.abs(stdDevIntrinsicsRO_75[4] / distortion_coefficientRO_75[0] * 100),
            distortion_coefficientRO[0] - distortion_coefficientRO_75[0],
        )
    )
    print(
        "k2 (RO): %+2.4f ±%2.4f (1-σ percent %2.1f %%) \t\t Delta k2 from initial calibration: %2.5f"
        % (
            distortion_coefficientRO_75[1],
            stdDevIntrinsicsRO_75[5],
            np.abs(stdDevIntrinsicsRO_75[5] / distortion_coefficientRO_75[1] * 100),
            distortion_coefficientRO[1] - distortion_coefficientRO_75[1],
        )
    )
    print(
        "k3 (RO): %+2.4f ±%2.4f (1-σ percent %2.1f %%) \t\t Delta k3 from initial calibration: %2.5f"
        % (
            distortion_coefficientRO_75[4],
            stdDevIntrinsicsRO_75[8],
            np.abs(stdDevIntrinsicsRO_75[8] / distortion_coefficientRO_75[4] * 100),
            distortion_coefficientRO[4] - distortion_coefficientRO_75[4],
        )
    )
    print(
        "p1 (RO): %+2.4f ±%2.4f (1-σ percent %2.1f %%) \t\t Delta p1 from initial calibration: %2.5f"
        % (
            distortion_coefficientRO_75[2],
            stdDevIntrinsicsRO_75[6],
            np.abs(stdDevIntrinsicsRO_75[6] / distortion_coefficientRO_75[2] * 100),
            distortion_coefficientRO[2] - distortion_coefficientRO_75[2],
        )
    )
    print(
        "p2 (RO): %+2.4f ±%2.4f (1-σ percent %2.1f %%) \t\t Delta p2 from initial calibration: %2.5f"
        % (
            distortion_coefficientRO_75[3],
            stdDevIntrinsicsRO_75[7],
            np.abs(stdDevIntrinsicsRO_75[7] / distortion_coefficientRO_75[3] * 100),
            distortion_coefficientRO[3] - distortion_coefficientRO_75[3],
        )
    )
    print("")
    print("Nr. of images: \t\t\t %05d" % len(objpoints_75))
    print("Nr. of chessboard points: \t %05d" % len(np.squeeze(objpoints_75[0])))
    print("")
    print("Original Camera Matrix: \n", K_initial)
    print("New Camera Matrix: \n", intrinsic_matrixRO_75)
    print("")
    print("Original Distortion Coefficients: \t", np.squeeze(D_initial))
    print("New Distortion Coefficients: \t\t", distortion_coefficientRO_75)
    print("")
    print("Writing lowest 75p perViewErrors CC to xml: %s" % savexml_file_best75p)
    cv_file = cv2.FileStorage(savexml_file_best75p, cv2.FILE_STORAGE_WRITE)
    cv_file.write("calibration_Time", str(datetime.datetime.now()))
    cv_file.write("image_Height", image_size[0])
    cv_file.write("image_Width", image_size[1])
    cv_file.write("Camera_Matrix", intrinsic_matrixRO_75)
    cv_file.write("Distortion_Coefficients", distortion_coefficientRO_75)
    cv_file.release()
    print("done.")

    return (
        objpoints,
        imgpoints,
        image_true_fn,
        image_size,
        rmsRO,
        intrinsic_matrixRO,
        distortion_coefficientRO,
        rvecsRO,
        tvecsRO,
    )


def calc_Fundamental_matrix(
    image_size, camA_imgpoints, K0, D0, camB_imgpoints, K1, D1, R, T
):
    # Q is disp_to_depth
    R0, R1, P0, P1, Q, roi_left, roi_right = cv2.stereoRectify(
        K0, D0, K1, D1, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )

    K0_optimal, camA_roi_optimal = cv2.getOptimalNewCameraMatrix(
        K0, D0, image_size, alpha=0.9
    )
    K1_optimal, camB_roi_optimal = cv2.getOptimalNewCameraMatrix(
        K1, D1, image_size, alpha=0.9
    )

    # Convert coordinates from chessboard to undistorted coordinates (faster than running the chessboard detection on image again and avoids resampling)
    # Using stereo calibration coefficients R0, P0, R1, P1
    camA_imgpoints_rectified = cv2.undistortPoints(
        np.squeeze(np.array(camA_imgpoints)), K0_optimal, D0, None, R0, P0
    )
    camA_imgpoints_rectified = np.squeeze(camA_imgpoints_rectified)
    camB_imgpoints_rectified = cv2.undistortPoints(
        np.squeeze(np.array(camB_imgpoints)), K1_optimal, D1, None, R1, P1
    )
    camB_imgpoints_rectified = np.squeeze(camB_imgpoints_rectified)

    F_all_chessboard, F_all_chessboard_mask = cv2.findFundamentalMat(
        camA_imgpoints_rectified,
        camB_imgpoints_rectified,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=0.5,
        confidence=3,
        maxIters=1000,
    )
    E_all_chessboard, E_all_chessboard_mask = cv2.findEssentialMat(
        camA_imgpoints_rectified,
        camB_imgpoints_rectified,
        K0_optimal,
        method=cv2.FM_RANSAC,
    )

    return (
        F_all_chessboard,
        F_all_chessboard_mask,
        E_all_chessboard,
        E_all_chessboard_mask,
        K0_optimal,
        K1_optimal,
    )


def stereo_CC_chessboard(
    save_stereo_file,
    image_size,
    objpoints,
    camA_imgpoints,
    camB_imgpoints,
    K0,
    D0,
    K1,
    D1,
):
    print(
        "Calibrating Stereo Camera and saving to XML. Using fixed intrinsic CC ... ",
        flush=True,
    )
    calib_flag = 0
    # calib_flag |= cv2.CALIB_USE_INTRINSIC_GUESS
    calib_flag |= cv2.CALIB_FIX_INTRINSIC
    # calib_flag |= cv2.CALIB_FIX_ASPECT_RATIO
    # stereoCalibrate(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2,
    # distCoeffs2, imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]) ->
    # retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F
    (stereo_CC_RMS, K0_new, D0_new, K1_new, D1_new, R, T, E, F) = cv2.stereoCalibrate(
        objpoints,
        camA_imgpoints,
        camB_imgpoints,
        K0,
        D0,
        K1,
        D1,
        image_size,
        flags=calib_flag,
        criteria=stereo_calib_criteria,
    )

    print("\n1. Stereo calibration (stereoCalibrate): ")
    print("Stereo calibration rms: ", stereo_CC_RMS)
    print("Stereo calibration T: ", T)
    print("Stereo calibration R: ", R)
    print("Stereo calibration F: ", F)
    print("Distance between cameras: ", np.linalg.norm(np.asarray(T)))

    # # stereoCalibrateExtended(objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1,
    # # cameraMatrix2, distCoeffs2, imageSize, R, T[, E[, F[, perViewErrors[, flags[, criteria]]]]]) ->
    # # retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, perViewErrors
    # #perViewErrors: Output vector of the RMS re-projection error estimated for each pattern view.
    R_initial = (
        R  # np.array([-0.03292444540920294, -0.04931732400513923, -1.5747491753532337])
    )
    T_initial = (
        T  # np.array([-0.26027954695229746, 0.11173310613979519, 0.8425073133847115])
    )
    (
        stereo_CC_RMS,
        K0_new,
        D0_new,
        K1_new,
        D1_new,
        R,
        T,
        E,
        F,
        stereo_CC_perViewErrors,
    ) = cv2.stereoCalibrateExtended(
        objpoints,
        camA_imgpoints,
        camB_imgpoints,
        K0,
        D0,
        K1,
        D1,
        image_size,
        R_initial,
        T_initial,
        flags=calib_flag,
        criteria=stereo_calib_criteria,
    )

    print(
        "\n2. Stereo calibration using R_initial and T_initial from first run (stereoCalibrateExtended):"
    )
    print("Stereo calibration rms: ", stereo_CC_RMS)
    print("Stereo calibration T: ", T)
    print("Stereo calibration R: ", R)
    print("Stereo calibration F: ", F)
    print("Distance between cameras: ", np.linalg.norm(np.asarray(T)))

    R_initial = (
        R  # np.array([-0.03292444540920294, -0.04931732400513923, -1.5747491753532337])
    )
    T_initial = (
        T  # np.array([-0.26027954695229746, 0.11173310613979519, 0.8425073133847115])
    )
    # Using only images with low errors: Discarding 25% of the images with highest perViewErrors
    nr_images = len(camA_imgpoints)
    nr_images75p = np.int16(np.round(nr_images * 0.75))
    camA_perViewErrors_idx = np.argsort(stereo_CC_perViewErrors[:, 0])
    if len(objpoints) > 1:
        objpoints_75plowestRMS = list(
            np.squeeze(
                np.array(objpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
            )
        )
        camA_imgpoints_75plowestRMS = list(
            np.squeeze(
                np.array(camA_imgpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
            )
        )
        camB_imgpoints_75plowestRMS = list(
            np.squeeze(
                np.array(camB_imgpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
            )
        )
    elif len(objpoints) == 1:
        objpoints_75plowestRMS = list(
            np.array(objpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
        )
        camA_imgpoints_75plowestRMS = list(
            np.array(camA_imgpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
        )
        camB_imgpoints_75plowestRMS = list(
            np.array(camB_imgpoints)[camA_perViewErrors_idx[0:nr_images75p], :, :]
        )

    (
        stereo_CC_RMS_75plowestRMS,
        K0_new_75plowestRMS,
        D0_new_75plowestRMS,
        K1_new_75plowestRMS,
        D1_new_75plowestRMS,
        R_75plowestRMS,
        T_75plowestRMS,
        E_75plowestRMS,
        F_75plowestRMS,
        stereo_CC_perViewErrors_75plowestRMS,
    ) = cv2.stereoCalibrateExtended(
        objpoints_75plowestRMS,
        camA_imgpoints_75plowestRMS,
        camB_imgpoints_75plowestRMS,
        K0,
        D0,
        K1,
        D1,
        image_size,
        R_initial,
        T_initial,
        flags=calib_flag,
        criteria=stereo_calib_criteria,
    )

    # R	Output rotation matrix. Together with the translation vector T, this matrix brings points given in the first camera's coordinate system to points in the second camera's coordinate system. In more technical terms, the tuple of R and T performs a change of basis from the first camera's coordinate system to the second camera's coordinate system. Due to its duality, this tuple is equivalent to the position of the first camera with respect to the second camera coordinate system.
    # T	Output translation vector, see description above.
    # E	Output essential matrix.
    # F	Output fundamental matrix.
    print(
        "\n3. Stereo calibration using R_initial and T_initial from second run and using %02d/%02d lowest-error images (stereoCalibrateExtended):"
        % (nr_images75p, nr_images)
    )
    print("Stereo calibration rms: ", stereo_CC_RMS_75plowestRMS)
    print("Stereo calibration T: ", T_75plowestRMS)
    print("Stereo calibration R: ", R_75plowestRMS)
    print("Stereo calibration F: ", F_75plowestRMS)
    print("Distance between cameras: ", np.linalg.norm(np.asarray(T_75plowestRMS)))

    # Will use results from images with lowest errors
    T = T_75plowestRMS
    R = R_75plowestRMS
    F = F_75plowestRMS
    # print('Initial K0 for Camera1:', K0)
    # print('New K0 for Camera1:', K0_new)
    # print('Initial K1 for Camera2:', K1)
    # print('New K1 for Camera2:', K1_new)
    # print('Initial D0 for Camera1:', D0)
    # print('New D0 for Camera1:', D0_new)
    # print('Initial D1 for Camera2:', D1)
    # print('New D1 for Camera2:', D1_new)

    R0, R1, P0, P1, Q, roi_left, roi_right = cv2.stereoRectify(
        K0, D0, K1, D1, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
    )
    # R1	Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    # R2	Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system.
    # P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
    # P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
    print("\nstereoRectify calibration R1: ", R0)
    print("stereoRectify calibration R2: ", R1)
    print("stereoRectify calibration P1: ", P0)
    print("stereoRectify calibration P2: ", P1)

    save_stereo_coefficients(
        save_stereo_file, K0, D0, K1, D1, R, T, E, F, R0, R1, P0, P1, Q
    )

    # cv_file = cv2.FileStorage(save_xml_file, cv2.FILE_STORAGE_WRITE)
    # cv_file.write("image_Height", iheight)
    # cv_file.write("image_Width", iwidth)
    # cv_file.write("Left_Camera_Matrix", K0)
    # cv_file.write("Left_Distortion_Coefficients", D0)
    # cv_file.write("Right_Camera_Matrix", K1)
    # cv_file.write("Right_Distortion_Coefficients", D1)
    # cv_file.write("Rotation_Matrix", R)
    # cv_file.write("Translation_Matrix", T)
    # cv_file.write("Essential_Matrix", E)
    # cv_file.write("Fundamental_Matrix", F)
    # cv_file.write("R1", R0)
    # cv_file.write("R2", R1)
    # cv_file.write("P1", P0)
    # cv_file.write("P2", P1)
    # cv_file.release()
    return (
        K0_new,
        D0_new,
        K1_new,
        D1_new,
        R,
        T,
        E,
        F,
        R0,
        R1,
        P0,
        P1,
        Q,
        roi_left,
        roi_right,
    )


if __name__ == "__main__":
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(
        description="Single Camera calibration with Chess and Charuco Boards using OpenCV V0.1 (bodo.bookhagen@uni-potsdam.de)"
    )

    parser.add_argument(
        "--camA_initial_CC",
        type=str,
        required=False,
        default="",
        help="Initial Camera Calibration (CC) in OpenCV XML format (e.g., taken from CalibIO: 'cam_A_calib_9parameters_fine_charuco_20Feb2022.xml'",
    )
    parser.add_argument(
        "--charuco_ifiles_camA",
        type=str,
        nargs="+",
        required=False,
        default="",
        help="Image files for charuco board calibration (e.g., 'sony_stereo_f13_iso1600/charuco/black_a_stereo/DSC*.JPG'",
    )
    parser.add_argument(
        "--camA_charuco_savexml_file",
        type=str,
        required=False,
        default="",
        help="OpenCV XML file with camera calibration from charuco board",
    )

    parser.add_argument(
        "--chess_ifiles_camA",
        type=str,
        nargs="+",
        required=False,
        default="",
        help="Image files for chess board calibration (e.g., 'sony_stereo_f13_iso1600/chess/black_a_stereo/DSC*.JPG'",
    )
    parser.add_argument(
        "--camA_chess_savexml_file",
        type=str,
        required=False,
        default="",
        help="OpenCV XML file with camera calibration from chessboard",
    )
    parser.add_argument(
        "--camA_chess_75pbest_savexml_file",
        type=str,
        required=False,
        default="",
        help="OpenCV XML file with camera calibration from chess board using only the best 75% of images and removing 25% of weakest images.",
    )

    parser.add_argument(
        "--camA_CC_comparison_3panel_png",
        default="",
        type=str,
        required=False,
        help="PNG output file showing results from CC for either charuco or chess board. 3 panel plot showing grid from each CC and their difference.",
    )
    parser.add_argument(
        "--camA_CC_comparison_1panel_png",
        default="",
        type=str,
        required=False,
        help="PNG output file showing results from CC for either charuco or chess board. 1 panel plot showing only difference grid.",
    )
    parser.add_argument(
        "--camA_Height",
        default=4000,
        type=int,
        required=False,
        help="Camera Height (y axis). 4000 pixels for Sony alpha-6 (24 MP), 5304 pixels for Sony alpha-7 (40 MP).",
    )
    parser.add_argument(
        "--camA_Width",
        default=6000,
        type=int,
        required=False,
        help="Camera Width (x axis). 6000 pixels for Sony alpha-6 (24 MP), 7952 pixels for Sony alpha-7 (40 MP).",
    )
    parser.add_argument(
        "--focal_length_pixels",
        default=9000.0,
        type=float,
        required=False,
        help="Focal length for camera calibration in pixels. Set to 9000 for Sony alpha-6 and 55 mm lense. Set to 12675 for Sony alpha-7 and 55 mm lense. Set to 18918 for Sony alpha-7 and 85 mm lense.",
    )

    args = parser.parse_args()

    # 1. Load existing Camera Calibration (calibrate camera and stereo rig separately)
    # args.camA_initial_CC='/home/bodo/Dropbox/QuadRig/20220330/single_calibration/camA_8parameters_charuco_calibio_30Mar2022.xml'
    # args.camA_Height=5304
    # args.camA_Width=7952
    # args.focal_length_pixels=9000
    H = args.camA_Height
    W = args.camA_Width
    imsize = np.array([W, H])
    F = args.focal_length_pixels
    cameraMatrixInit = np.array(
        [[F, 0.0, imsize[0] / 2.0], [0.0, F, imsize[1] / 2.0], [0.0, 0.0, 1.0]]
    )
    D = np.zeros((5, 1))

    if not args.camA_initial_CC == "":
        print("Load CC from %s" % args.camA_initial_CC)
        K0, D0, H0, W0 = load_CC_from_XML(args.camA_initial_CC)
    elif args.camA_initial_CC == "":
        K0 = cameraMatrixInit
        D0 = D
        H0 = H
        W0 = W

    # 2a. Find charuco markers and image coordinates using existing CC
    if not args.charuco_ifiles_camA == "":
        dirname = "/".join(
            os.path.dirname(args.charuco_ifiles_camA[0]).split("/")[0:-1]
        )
        (
            camA_corners_charuco,
            camA_ids_charuco,
            camA_imsize_charuco,
            camA_CC_rms_charuco,
            camA_camera_matrix_charuco,
            camA_distortion_coefficients_charuco,
            camA_rotation_vectors_charuco,
            camA_translation_vectors_charuco,
            camA_CHARUCO_BOARD,
        ) = CC_charucoboard(
            args.charuco_ifiles_camA,
            K0,
            D0,
            W0,
            H0,
            savepng_output=True,
            savepng_corners_output_file="camA_charuco_",
            savepng_path=dirname,
            CHARUCOBOARD_ROWCOUNT=18,
            CHARUCOBOARD_COLCOUNT=25,
            square_size_m=0.015,
            marker_length_m=0.012,
            CHARUCO_DICT=cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000),
        )
        save_opencv_CC_xml(
            args.camA_charuco_savexml_file,
            camA_imsize_charuco,
            camA_camera_matrix_charuco,
            camA_distortion_coefficients_charuco,
        )
        if args.camA_initial_CC == "":
            # ONLY if empty: Set initial camera Matrix and Distortion coefficient to results from charuco file
            K0 = camA_camera_matrix_charuco
            D0 = camA_distortion_coefficients_charuco
        allCorners_sum = sum([len(listElem) for listElem in camA_corners_charuco])
        plot_img_corners(
            camA_corners_charuco,
            png_fname=os.path.join(
                dirname, "cam_charuco_%03dcorners.png" % allCorners_sum
            ),
            title="Cam Charuco corners (%d) from %02d images"
            % (allCorners_sum, len(camA_corners_charuco)),
            w=W,
            h=H,
            fontsize=10,
        )

    # could add gridding here - create an equally-spaced grid with 10 pixels or so and use this for calibration

    # 2b. Find chessboard markers
    # args.chess_ifiles_camA='/home/bodo/20220701/images/sony_a7_single/55mm/chess/near/DSC*.JPG'
    # args.camA_chess_savexml_file='/home/bodo/20220701/images/sony_a7_single/55mm/chess/sony_a7_55m_chess.xml'
    # args.camA_chess_75plowest_savexml_file='/home/bodo/20220701/images/sony_a7_single/55mm/chess/sony_a7_55m_chess_75p.xml'
    chessboard_width = 29 - 1
    chessboard_height = 18 - 1
    dirname = "/".join(os.path.dirname(args.chess_ifiles_camA[0]).split("/")[0:-1])
    (
        camA_objpoints_chess,
        camA_imgpoints_chess,
        camA_image_true_fn_chess,
        camA_image_size_chess,
        camA_rmsRO_chess,
        camA_camera_matrix_chess,
        camA_distortion_coefficients_chess,
        camA_rvecs_chess,
        camA_tvecs_chess,
    ) = CC_chessboard(
        args.chess_ifiles_camA,
        K0.copy(),
        D0.copy(),
        W0,
        H0,
        chessboard_width=chessboard_width,
        chessboard_height=chessboard_height,
        savepng_output=True,
        savepng_corners_output_file="camA_chessboard_",
        savepng_path=dirname,
        savexml_file=args.camA_chess_savexml_file,
        savexml_file_best75p=args.camA_chess_75pbest_savexml_file,
    )
    allCorners_sum = sum([len(listElem) for listElem in camA_imgpoints_chess])
    plot_img_corners(
        camA_imgpoints_chess,
        png_fname=os.path.join(dirname, "camA_chess_%03dcorners.png" % allCorners_sum),
        title="CamA Chess corners (%d) from %02d images"
        % (allCorners_sum, len(camA_imgpoints_chess)),
        w=W0,
        h=H0,
        fontsize=10,
    )

    # 3. Plot Camera Calibration comparison
    if len(args.camA_initial_CC) == 0 and not args.charuco_ifiles_camA == "":
        title0 = "CamA Charuco CC from OpenCV"
        visualizeDistortion(
            args.camA_CC_comparison_1panel_png,
            camA_camera_matrix_charuco,
            camA_distortion_coefficients_charuco,
            imsize[1],
            imsize[0],
            fontsize=10,
            contourLevels=10,
            nstep=100,
            title0=title0,
        )
    elif len(args.camA_initial_CC) == 0 and not args.chess_ifiles_camA == "":
        title0 = "CamA Chess CC from OpenCV"
        visualizeDistortion(
            args.camA_CC_comparison_1panel_png,
            camA_camera_matrix_chess,
            camA_distortion_coefficients_chess,
            imsize[1],
            imsize[0],
            fontsize=10,
            contourLevels=10,
            nstep=100,
            title0=title0,
        )
    elif len(args.camA_initial_CC) > 0 and not args.charuco_ifiles_camA == "":
        title0 = "CamA Charuco initial CC"
        title1 = "CamA Charuco CC from OpenCV"
        title_diff = "CC Difference between Calib.io and OpenCV (Charuco)"
        visualizeDistortion_comparison_2(
            args.camA_CC_comparison_3panel_png,
            args.camA_CC_comparison_1panel_png,
            K0,
            D0,
            camA_camera_matrix_charuco,
            camA_distortion_coefficients_charuco,
            imsize[1],
            imsize[0],
            fontsize=10,
            contourLevels=10,
            nstep=100,
            title0=title0,
            title1=title1,
            title_diff=title_diff,
        )
    elif len(args.camA_initial_CC) > 0 and not args.chess_ifiles_camA == "":
        title0 = "CamA Chess initial CC"
        title1 = "CamA Chess CC from OpenCV"
        title_diff = "CC Difference between initial and OpenCV (Chess)"
        visualizeDistortion_comparison_2(
            args.camA_CC_comparison_3panel_png,
            args.camA_CC_comparison_1panel_png,
            K0,
            D0,
            camA_camera_matrix_chess,
            camA_distortion_coefficients_chess,
            imsize[1],
            imsize[0],
            fontsize=10,
            contourLevels=10,
            nstep=100,
            title0=title0,
            title1=title1,
            title_diff=title_diff,
        )
