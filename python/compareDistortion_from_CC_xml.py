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


def visualizeDistortion_comparison_2(save_fname_3panel, save_fname_1panel, K0, D0, K1, D1, h, w,
        fontsize=16, contourLevels=10, nstep=20, title0='cam0',
        title1='cam1', title_diff='Difference'):
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
    xyz = np.linalg.lstsq(K0, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = K0[0,0]*xpp + K0[0,2]
    v2 = K0[1,1]*ypp + K0[1,2]
    du0 = np.copy(u2.ravel() - u.ravel())
    dv0 = np.copy(v2.ravel() - v.ravel())
    dr0 = np.copy(np.hypot(du0,dv0).reshape(u.shape))

    fig, ax = plt.subplots(1,3,figsize=(12, 8),dpi=600)
    ax[0].quiver(u.ravel(), v.ravel(), du0, -dv0, color="dodgerblue")
    ax[0].plot(w//2, h//2, "x", K0[0,2], K0[1,2],"^", markersize=fontsize)
    CS = ax[0].contour(u, v, dr0, colors="black", levels=contourLevels)
    ax[0].set_aspect('equal', 'box')
    ax[0].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[0].set_title(title0, fontsize=fontsize)
    ax[0].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[0].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[0].set_ylim(max(v.ravel()),0)

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
    xyz = np.linalg.lstsq(K1, b,rcond=None)[0]
    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3
    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = K1[0,0]*xpp + K1[0,2]
    v2 = K1[1,1]*ypp + K1[1,2]
    du1 = np.copy(u2.ravel() - u.ravel())
    dv1 = np.copy(v2.ravel() - v.ravel())
    dr1 = np.copy(np.hypot(du1,dv1).reshape(u.shape))

    ax[1].quiver(u.ravel(), v.ravel(), du1, -dv1, color="dodgerblue")
    ax[1].plot(w//2, h//2, "x", K1[0,2], K1[1,2],"^", markersize=fontsize)
    CS = ax[1].contour(u, v, dr1, colors="black", levels=contourLevels)
    ax[1].set_aspect('equal', 'box')
    ax[1].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[1].set_title(title1, fontsize=fontsize)
    ax[1].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[1].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[1].set_ylim(max(v.ravel()),0)

    ax[2].quiver(u.ravel(), v.ravel(), (du1-du0) , (-dv1+dv0) , color="dodgerblue")
    ax[2].plot(w//2, h//2, "x", K0[0,2], K0[1,2],"o", K1[0,2], K1[1,2],"+", markersize=fontsize)
    CS = ax[2].contour(u, v, dr1-dr0, colors="black", levels=contourLevels)
    ax[2].set_aspect('equal', 'box')
    ax[2].clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax[2].set_title(title_diff, fontsize=fontsize)
    ax[2].set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax[2].set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
    ax[2].set_ylim(max(v.ravel()),0)
    fig.tight_layout()
    fig.savefig(save_fname_3panel, dpi=300)
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12, 8),dpi=300)
    ax.quiver(u.ravel(), v.ravel(), (du1-du0) , (-dv1+dv0) , color="dodgerblue")
    ax.plot(w//2, h//2, "x", K0[0,2], K0[1,2],"o", K1[0,2], K1[1,2],"+", markersize=fontsize)
    CS = ax.contour(u, v, dr1-dr0, colors="black", levels=contourLevels)
    ax.set_aspect('equal', 'box')
    ax.clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax.set_title(title_diff, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)"%w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)"%h, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylim(max(v.ravel()),0)
    fig.tight_layout()
    fig.savefig(save_fname_1panel, dpi=300)
    plt.close()

if __name__ == '__main__':
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(description='Compare two Camera Calibration Files in OpenCV Format and plot difference as vector field (bodo.bookhagen@uni-potsdam.de)')
    parser.add_argument('--save_fname_3panel', type=str, required=True, help="Outputfile showing calibration field 1, 2, and their differences (PNG, 300 dpi)")
    parser.add_argument('--save_fname_1panel', type=str, required=True, help="Outputfile showing difference of vector field (PNG, 300 dpi)")
    parser.add_argument('--CC0_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title0', type=str, required=True, help='Title for plot')
    parser.add_argument('--CC1_fn', type=str, required=True, help="OpenCV CameraCalibration XML file")
    parser.add_argument('--title1', type=str, required=True, help='Title for plot')
    parser.add_argument('--title_diff', type=str, required=True, help='Title for plot showing difference between CC')
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

    visualizeDistortion_comparison_2(args.save_fname_3panel, args.save_fname_1panel,
        K0, D0, K1, D1, args.h, args.w,
        fontsize=10, contourLevels=10, nstep=100, title0=args.title0,
        title1=args.title1, title_diff=args.title_diff)
