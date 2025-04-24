#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import tqdm
from matplotlib.path import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    imgpoints = [c.reshape(-1,2) for c in imgpoints]
    mean_error = None
    error_x = []
    error_y = []
    rms = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = imgpoints2.reshape(-1,2)

        # if not all markers were found, then the norm below will fail
        if len(imgpoints[i]) != len(imgpoints2):
            continue

        error_x.append(list(imgpoints2[:,0] - imgpoints[i][:,0]))
        error_y.append(list(imgpoints2[:,1] - imgpoints[i][:,1]))

        rr = np.sum((imgpoints2 - imgpoints[i])**2, axis=1)
        if mean_error is None:
            mean_error = rr
        else:
            mean_error = np.hstack((mean_error, rr))
        rr = np.sqrt(np.mean(rr))
        rms.append(rr)

    m_error = np.sqrt(np.mean(mean_error))
    return m_error, rms, [error_x, error_y]


def visualizeReprojErrors(totalRSME, rmsPerView, reprojErrs, fontSize=12,legend=False,xlim=None,ylim=None):
    fig, ax = plt.subplots()
    for i,(rms,x,y) in enumerate(zip(rmsPerView,reprojErrs[0],reprojErrs[1])):
        ax.scatter(x,y,label=f"RMSE [{i}]: {rms:0.3f}")

    # change dimensions if legend displayed
    if legend:
        ax.legend(fontsize=fontSize)
        ax.axis('equal')
    else:
        ax.set_aspect('equal', 'box')
    ax.set_title(f"Reprojection Error (in pixels), RMSE: {totalRSME:0.4f}", fontsize=fontSize)
    ax.set_xlabel("X", fontsize=fontSize)
    ax.set_ylabel("Y", fontsize=fontSize)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=fontSize)
    ax.grid(True);
    plt.show()


def visualizeDistortion(save_fname, K, D, h, w, fontsize=16, contourLevels=10, nstep=20, title_string='Distortion Model'):
    """
    taken from: https://github.com/MomsFriendlyRobotCompany/opencv_camera/blob/master/opencv_camera/distortion.py
    and following:

    http://amroamroamro.github.io/mexopencv/opencv/calibration_demo.html
    https://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#details
    """
    # K = [fx 0 cx; 0 fy cy; 0 0 1]
    #
    # * focal lengths   : fx, fy
    # * aspect ratio    : a = fy/fx
    # * principal point : cx, cy
    M = K
    # D = [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]
    #
    # * radial distortion     : k1, k2, k3
    # * tangential distortion : p1, p2
    # * rational distortion   : k4, k5, k6
    # * thin prism distortion : s1, s2, s3, s4
    # * tilted distortion     : taux, tauy (ignored here)
    #
    D = D.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]

    # nstep = 20
    u,v = np.meshgrid(
        np.arange(0, w, nstep),
        np.arange(0, h, nstep)
    )

    b = np.array([
        u.ravel(),
        v.ravel(),
        np.ones(u.size)
    ])

    xyz = np.linalg.lstsq(M, b,rcond=None)[0]

    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3

    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)

    fig, ax = plt.subplots()
    ax.quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax.plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax.contour(u, v, dr, colors="black", levels=contourLevels)
    ax.set_aspect('equal', 'box')
    ax.clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax.set_title(title_string, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylim(max(v.ravel()),0)
    fig.tight_layout()
    fig.savefig(save_fname, dpi=600)
    plt.close()

def visualizeDistortion_4(save_fname, K0, D0, K1, D1, K2, D2, K3, D3, h, w,
    fontsize=16, contourLevels=10, nstep=20, title0='Initial Parameters from CC file',
    title1='RO Calibration Parameters (no pre-calibration)',
    title2='RO Calibration Parameters (with CC file)',
    title4='Lowest RMS Calibration Parameters (with CC file)'):
    M = K0
    D = D0.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u,v = np.meshgrid(np.arange(0, w, nstep),np.arange(0, h, nstep))
    b = np.array([u.ravel(),v.ravel(),np.ones(u.size)])
    xyz = np.linalg.lstsq(M, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)

    fig, ax = plt.subplots(2,2,figsize=(12, 8),dpi=600)
    ax[0,0].quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax[0,0].plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax[0,0].contour(u, v, dr, colors="black", levels=contourLevels)
    ax[0,0].set_aspect('equal', 'box')
    ax[0,0].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[0,0].set_title(title0, fontsize=fontsize)
    ax[0,0].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[0,0].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[0,0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0,0].set_ylim(max(v.ravel()),0)

    M = K1
    D = D1.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u,v = np.meshgrid(np.arange(0, w, nstep),np.arange(0, h, nstep))
    b = np.array([u.ravel(),v.ravel(),np.ones(u.size)])
    xyz = np.linalg.lstsq(M, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)
    ax[0,1].quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax[0,1].plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax[0,1].contour(u, v, dr, colors="black", levels=contourLevels)
    ax[0,1].set_aspect('equal', 'box')
    ax[0,1].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[0,1].set_title(title1, fontsize=fontsize)
    ax[0,1].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[0,1].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[0,1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0,1].set_ylim(max(v.ravel()),0)

    M = K2
    D = D2.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u,v = np.meshgrid(np.arange(0, w, nstep),np.arange(0, h, nstep))
    b = np.array([u.ravel(),v.ravel(),np.ones(u.size)])
    xyz = np.linalg.lstsq(M, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)
    ax[1,0].quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax[1,0].plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax[1,0].contour(u, v, dr, colors="black", levels=contourLevels)
    ax[1,0].set_aspect('equal', 'box')
    ax[1,0].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[1,0].set_title(title2, fontsize=fontsize)
    ax[1,0].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[1,0].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[1,0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1,0].set_ylim(max(v.ravel()),0)

    M = K3
    D = D3.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]
    u,v = np.meshgrid(np.arange(0, w, nstep),np.arange(0, h, nstep))
    b = np.array([u.ravel(),v.ravel(),np.ones(u.size)])
    xyz = np.linalg.lstsq(M, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)
    ax[1,1].quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax[1,1].plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax[1,1].contour(u, v, dr, colors="black", levels=contourLevels)
    ax[1,1].set_aspect('equal', 'box')
    ax[1,1].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[1,1].set_title(title4, fontsize=fontsize)
    ax[1,1].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[1,1].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[1,1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1,1].set_ylim(max(v.ravel()),0)

    fig.tight_layout()
    fig.savefig(save_fname, dpi=600)
    plt.close()

if __name__ == '__main__':
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Visualize up to 4 distortion patterns from camera calibration files (bodo.bookhagen@uni-potsdam.de)')
    parser.add_argument('--png_fn', type=str, required=True, help="Outputfile for PNG (600 dpi)")
    parser.add_argument('--CC0_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title0', type=str, required=True, help='Title for plot')
    parser.add_argument('--CC1_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title1', type=str, required=True, help='Title for plot')
    parser.add_argument('--CC2_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title2', type=str, required=True, help='Title for plot')
    parser.add_argument('--CC3_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title3', type=str, required=True, help='Title for plot')
    parser.add_argument('--h', type=int, default=4000, required=False, help='Image height (pixels)')
    parser.add_argument('--w', type=int, default=6000, required=False, help='Image width (pixels)')
    args = parser.parse_args()

    if os.path.exists(args.CC0_fn):
        cv_file = cv2.FileStorage(args.CC0_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC0_fn, cv2.FILE_STORAGE_READ)
        K0 = np.squeeze(cv_file.getNode('Camera_Matrix').mat())
        D0 = np.squeeze(cv_file.getNode('Distortion_Coefficients').mat())
        # h0 = np.squeeze(cv_file.getNode('image_Height'))
        # w0 = np.squeeze(cv_file.getNode('image_Width'))
        cv_file.release()

    if os.path.exists(args.CC1_fn):
        cv_file = cv2.FileStorage(args.CC1_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC1_fn, cv2.FILE_STORAGE_READ)
        K1 = np.squeeze(cv_file.getNode('Camera_Matrix').mat())
        D1 = np.squeeze(cv_file.getNode('Distortion_Coefficients').mat())
        # h1 = np.squeeze(cv_file.getNode('image_Height'))
        # w1 = np.squeeze(cv_file.getNode('image_Width'))
        cv_file.release()

    if os.path.exists(args.CC2_fn):
        cv_file = cv2.FileStorage(args.CC2_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC2_fn, cv2.FILE_STORAGE_READ)
        K2 = np.squeeze(cv_file.getNode('Camera_Matrix').mat())
        D2 = np.squeeze(cv_file.getNode('Distortion_Coefficients').mat())
        # h2 = np.squeeze(cv_file.getNode('image_Height'))
        # w2 = np.squeeze(cv_file.getNode('image_Width'))
        cv_file.release()

    if os.path.exists(args.CC3_fn):
        cv_file = cv2.FileStorage(args.CC3_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC3_fn, cv2.FILE_STORAGE_READ)
        K3 = np.squeeze(cv_file.getNode('Camera_Matrix').mat())
        D3 = np.squeeze(cv_file.getNode('Distortion_Coefficients').mat())
        # h3 = np.squeeze(cv_file.getNode('image_Height'))
        # w3 = np.squeeze(cv_file.getNode('image_Width'))
        cv_file.release()

    # if h0 == h1 and w0 == w1:
    #     if h0 == h2 and w0 == w2:
    #         if h0 == h3 and w0 == w3:
    #             print('All CC files have same image height and width.')
    #         else:
    #             print('Image height and width of CC files 0 and 3 need to be the same.\nh0=%d, h1=%d, w0=%d, w1=%d'%(h0,h3,w0,w3))
    #             exit(-1)
    #     else:
    #         print('Image height and width of CC files 0 and 2 need to be the same.\nh0=%d, h1=%d, w0=%d, w1=%d'%(h0,h2,w0,w2))
    #         exit(-1)
    # else:
    #     # print('Image height and width of CC files 0 and 1 need to be the same.\nh0=%d, h1=%d, w0=%d, w1=%d'%(h0,h1,w0,w1))
    #     print(h0)
    #     print(h1)
    #     exit(-1)
    # h = h0
    # w = w0

    visualizeDistortion_4(args.png_fn, K0, D0, K1, D1, K2, D2, K3, D3,
        args.h, args.w, fontsize=10, contourLevels=10, nstep=100,
        title0=args.title0,
        title1=args.title1,
        title2=args.title2,
        title4=args.title3)
