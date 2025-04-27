#!/usr/bin/env python
import cv2
import numpy as np
import scipy.ndimage
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.colors as colors
import xml.etree.ElementTree as ET


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
    cfontsize=14,
    contourLevels=5,
    nstep=50,
    clinewidth=1,
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
    xpp = xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    ypp = (
        yp * coef + p1 * (r2 + 2 * yp**2) + 2 * p2 * (xp * yp) + D[11] * r2 + D[11] * r4
    )
    u2 = K0[0, 0] * xpp + K0[0, 2]
    v2 = K0[1, 1] * ypp + K0[1, 2]
    du0 = np.copy(u2.ravel() - u.ravel())
    dv0 = np.copy(v2.ravel() - v.ravel())
    dr0 = np.copy(np.hypot(du0, dv0).reshape(u.shape))

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), dpi=600, sharey=True)
    ax[0].quiver(u.ravel(), v.ravel(), du0, -dv0, color="dodgerblue")
    ax[0].plot(w // 2, h // 2, "x", K0[0, 2], K0[1, 2], "^", markersize=fontsize)
    CS = ax[0].contour(
        u, v, dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    ax[0].set_aspect("equal", "box")
    ax[0].clabel(CS, inline=1, fontsize=cfontsize, fmt="%0.0f")
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
    xpp = xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    ypp = (
        yp * coef + p1 * (r2 + 2 * yp**2) + 2 * p2 * (xp * yp) + D[11] * r2 + D[11] * r4
    )
    u2 = K1[0, 0] * xpp + K1[0, 2]
    v2 = K1[1, 1] * ypp + K1[1, 2]
    du1 = np.copy(u2.ravel() - u.ravel())
    dv1 = np.copy(v2.ravel() - v.ravel())
    dr1 = np.copy(np.hypot(du1, dv1).reshape(u.shape))

    ax[1].quiver(u.ravel(), v.ravel(), du1, -dv1, color="dodgerblue")
    ax[1].plot(w // 2, h // 2, "x", K1[0, 2], K1[1, 2], "^", markersize=fontsize)
    CS = ax[1].contour(
        u, v, dr1, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    ax[1].set_aspect("equal", "box")
    ax[1].clabel(CS, inline=1, fontsize=cfontsize, fmt="%0.0f")
    ax[1].set_title(title1, fontsize=fontsize)
    ax[1].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    # ax[1].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax[1].tick_params(axis="both", which="major", labelsize=fontsize)
    ax[1].set_ylim(max(v.ravel()), 0)

    ax[2].quiver(u.ravel(), v.ravel(), (du1 - du0), (-dv1 + dv0), color="darkred")
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
    CS = ax[2].contour(
        u, v, dr1 - dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    ax[2].set_aspect("equal", "box")
    ax[2].clabel(CS, inline=1, fontsize=cfontsize, fmt="%0.0f")
    ax[2].set_title(title_diff, fontsize=fontsize)
    ax[2].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    # ax[2].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax[2].tick_params(axis="both", which="major", labelsize=fontsize)
    ax[2].set_ylim(max(v.ravel()), 0)
    fig.tight_layout()
    fig.savefig(save_fname_3panel, dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    ax.quiver(u.ravel(), v.ravel(), (du1 - du0), (-dv1 + dv0), color="darkred")
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
    CS = ax.contour(
        u, v, dr1 - dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    ax.set_aspect("equal", "box")
    ax.clabel(CS, inline=1, fontsize=cfontsize, fmt="%0.0f")
    # clabel.text.set_fontweight("bold")
    ax.set_title(title_diff, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_ylim(max(v.ravel()), 0)
    fig.tight_layout()
    fig.savefig(save_fname_1panel, dpi=300)
    plt.close()


def visualizeDistortion_comparison_2image(
    save_fname_2panel,
    save_fname_2panel_diff,
    save_fname_1panel,
    K0,
    D0,
    K1,
    D1,
    h,
    w,
    fontsize=16,
    cfontsize=14,
    contourLevels=5,
    nstep=10,
    clinewidth=1,
    arrowmarkersize=10,
    title0="cam0",
    title1="cam1",
    title_diff="Difference",
    suptitle="Suptitle",
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
    xpp = xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    ypp = (
        yp * coef + p1 * (r2 + 2 * yp**2) + 2 * p2 * (xp * yp) + D[11] * r2 + D[11] * r4
    )
    u2 = K0[0, 0] * xpp + K0[0, 2]
    v2 = K0[1, 1] * ypp + K0[1, 2]
    du0 = np.copy(u2.ravel() - u.ravel())
    dv0 = np.copy(v2.ravel() - v.ravel())
    dr0 = np.copy(np.hypot(du0, dv0).reshape(u.shape))
    dr0r = scipy.ndimage.zoom(dr0, h / dr0.shape[0], order=1)

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
    xpp = xp * coef + 2 * p1 * (xp * yp) + p2 * (r2 + 2 * xp**2) + D[8] * r2 + D[9] * r4
    ypp = (
        yp * coef + p1 * (r2 + 2 * yp**2) + 2 * p2 * (xp * yp) + D[11] * r2 + D[11] * r4
    )
    u2 = K1[0, 0] * xpp + K1[0, 2]
    v2 = K1[1, 1] * ypp + K1[1, 2]
    du1 = np.copy(u2.ravel() - u.ravel())
    dv1 = np.copy(v2.ravel() - v.ravel())
    dr1 = np.copy(np.hypot(du1, dv1).reshape(u.shape))
    dr1r = scipy.ndimage.zoom(dr1, h / dr1.shape[0], order=1)

    axLeft_vmin = np.min(np.array([np.percentile(dr0r, 1), np.percentile(dr1r, 1)]))
    axLeft_vmax = np.max(np.array([np.percentile(dr0r, 99), np.percentile(dr1r, 99)]))
    fig = plt.figure(layout="constrained", figsize=(16, 6), dpi=300)
    subfigs = fig.subfigures(1, 1)
    # fig, ax = plt.subplots(1, 3, figsize=(16, 6), dpi=600, sharey=True)
    axLeft = subfigs.subplots(1, 2, sharey=True)
    axLeft[0].quiver(
        u.ravel(), v.ravel(), du0, -dv0, color="white", linewidths=arrowmarkersize
    )
    axLeft[0].plot(
        w // 2,
        h // 2,
        "x",
        K0[0, 2],
        K0[1, 2],
        "^",
        markersize=fontsize / 2,
    )
    im0 = axLeft[0].imshow(
        dr0r,
        cmap="viridis",
        vmin=axLeft_vmin,
        vmax=axLeft_vmax,
        alpha=0.5,
    )
    CS = axLeft[0].contour(
        u, v, dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    axLeft[0].set_aspect("equal", "box")
    axLeft[0].clabel(CS, inline=1, colors="black", fontsize=cfontsize, fmt="%0.0f")
    axLeft[0].set_title(title0, fontsize=fontsize)
    axLeft[0].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    axLeft[0].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    axLeft[0].tick_params(axis="both", which="major", labelsize=fontsize)
    axLeft[0].set_ylim(max(v.ravel()), 0)

    axLeft[1].quiver(
        u.ravel(), v.ravel(), du1, -dv1, color="white", linewidths=arrowmarkersize
    )
    axLeft[1].plot(
        w // 2, h // 2, "x", K1[0, 2], K1[1, 2], "^", markersize=fontsize / 2
    )
    CS = axLeft[1].contour(
        u, v, dr1, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    im1 = axLeft[1].imshow(
        dr1r,
        cmap="viridis",
        vmin=axLeft_vmin,
        vmax=axLeft_vmax,
        alpha=0.5,
    )
    axLeft[1].set_aspect("equal", "box")
    axLeft[1].clabel(CS, inline=1, fontsize=cfontsize, colors="black", fmt="%0.0f")
    axLeft[1].set_title(title1, fontsize=fontsize)
    axLeft[1].set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    axLeft[1].tick_params(axis="both", which="major", labelsize=fontsize)
    axLeft[1].set_ylim(max(v.ravel()), 0)
    clb_h0 = subfigs.colorbar(im0, shrink=0.8, ax=axLeft, location="bottom")
    clb_h0.set_label("Distortion offset [pixel]")
    fig.suptitle(suptitle, fontsize=fontsize + 4)
    fig.savefig(save_fname_2panel, dpi=300)
    plt.close()

    diff_vmin = np.min(
        np.array(
            [
                np.abs(np.percentile(dr1r - dr0r, 1)),
                np.abs(np.percentile(dr1r - dr0r, 99)),
            ]
        )
    )
    diff_vmax = np.max(
        np.array(
            [
                np.abs(np.percentile(dr1r - dr0r, 1)),
                np.abs(np.percentile(dr1r - dr0r, 99)),
            ]
        )
    )
    diff_v = np.max(np.array([diff_vmin, diff_vmax]))
    fig = plt.figure(layout="constrained", figsize=(16, 6), dpi=600)
    subfigs = fig.subfigures(1, 1)
    (axL, axR) = subfigs.subplots(1, 2, sharey=True)
    axL.quiver(
        u.ravel(),
        v.ravel(),
        (du1 - du0),
        (-dv1 + dv0),
        color="black",
        linewidths=arrowmarkersize,
    )
    axL.plot(
        w // 2,
        h // 2,
        "x",
        K0[0, 2],
        K0[1, 2],
        "o",
        K1[0, 2],
        K1[1, 2],
        "+",
        markersize=fontsize / 2,
    )
    CS = axL.contour(
        u, v, dr1 - dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    im2 = axL.imshow(
        dr1r - dr0r,
        cmap="PiYG",
        vmin=-diff_v,
        vmax=diff_v,
        alpha=0.5,
    )
    axL.set_aspect("equal", "box")
    axL.clabel(CS, inline=1, fontsize=cfontsize, colors="black", fmt="%0.0f")
    axL.set_title(title_diff, fontsize=fontsize)
    axL.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    axL.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    axL.tick_params(axis="both", which="major", labelsize=fontsize)
    axL.set_ylim(max(v.ravel()), 0)
    clb_h1 = subfigs.colorbar(im2, shrink=0.8, ax=axL, location="bottom")
    clb_h1.set_label(r"$\Delta$ Distortion offset [pixel]")

    np.seterr(divide="ignore", invalid="ignore")
    diff_vminp = np.nanpercentile((dr0r - dr1r) / dr0r, 1)
    diff_vmaxp = np.nanpercentile((dr0r - dr1r) / dr0r, 99)
    diff_vp = np.max(np.array([diff_vminp, diff_vmaxp]))
    axR.quiver(
        u.ravel(),
        v.ravel(),
        (du1 - du0),
        (-dv1 + dv0),
        color="black",
        linewidths=arrowmarkersize,
    )
    axR.plot(
        w // 2,
        h // 2,
        "x",
        K0[0, 2],
        K0[1, 2],
        "o",
        K1[0, 2],
        K1[1, 2],
        "+",
        markersize=fontsize / 2,
    )
    CS = axR.contour(
        u,
        v,
        ((dr0 - dr1) / dr0) * 100,
        colors="black",
        levels=contourLevels,
        linewidths=clinewidth,
    )
    im3 = axR.imshow(
        np.abs((dr0r - dr1r) / dr0r),
        cmap="plasma",
        norm=colors.LogNorm(vmin=np.abs(diff_vminp), vmax=np.abs(diff_vmaxp)),
        alpha=0.5,
    )
    axR.set_aspect("equal", "box")
    axR.clabel(CS, inline=1, fontsize=cfontsize, colors="black", fmt="%0.0f")
    axR.set_title(title_diff, fontsize=fontsize)
    axR.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    # ax[2].set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    axR.tick_params(axis="both", which="major", labelsize=fontsize)
    axR.set_ylim(max(v.ravel()), 0)
    clb_h2 = subfigs.colorbar(im3, shrink=0.8, ax=axR, location="bottom")
    clb_h2.set_label(
        r"Abs. $\Delta$ Distortion offset normalized by first CC value [factor]"
    )
    fig.suptitle(suptitle, fontsize=fontsize + 4)
    fig.savefig(save_fname_2panel_diff, dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    ax.quiver(
        u.ravel(),
        v.ravel(),
        (du1 - du0),
        (-dv1 + dv0),
        color="black",
        linewidths=arrowmarkersize,
    )
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
        markersize=fontsize / 2,
    )
    CS = ax.contour(
        u, v, dr1 - dr0, colors="black", levels=contourLevels, linewidths=clinewidth
    )
    im = ax.imshow(
        dr1r - dr0r,
        cmap="PiYG",
        vmin=-diff_v,
        vmax=diff_v,
        alpha=0.5,
    )
    ax.set_aspect("equal", "box")
    ax.clabel(CS, inline=1, colors="black", fontsize=cfontsize, fmt="%0.0f")
    # clabel.text.set_fontweight("bold")
    ax.set_title(title_diff, fontsize=fontsize)
    ax.set_xlabel("u (along X axis with %d pixels)" % w, fontsize=fontsize)
    ax.set_ylabel("v (along Y axis with %d pixels)" % h, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_ylim(max(v.ravel()), 0)
    clb_h = plt.colorbar(im, orientation="horizontal")
    clb_h.set_label(r"$\Delta$ Distortion offset [pixel]")

    fig.tight_layout()
    fig.savefig(save_fname_1panel, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Check the help parameters to understand arguments
    parser = argparse.ArgumentParser(
        description="Compare two Camera Calibration Files in OpenCV Format and plot difference as vector field (bodo.bookhagen@uni-potsdam.de)"
    )
    parser.add_argument(
        "--save_fname_2panel",
        type=str,
        required=True,
        help="Outputfile showing calibration fields 1 and 2. (PNG, 300 dpi)",
    )
    parser.add_argument(
        "--save_fname_2panel_diff",
        type=str,
        required=True,
        help="Outputfile showing absolute difference of calibration fields 1 and 2. (PNG, 300 dpi)",
    )
    parser.add_argument(
        "--save_fname_1panel",
        type=str,
        required=True,
        help="Outputfile showing difference of vector field (PNG, 300 dpi)",
    )
    parser.add_argument(
        "--CC0_fn", type=str, required=True, help="OpenCV CameraCalibration XML file"
    )
    parser.add_argument("--title0", type=str, required=True, help="Title for plot")
    parser.add_argument(
        "--CC1_fn", type=str, required=True, help="OpenCV CameraCalibration XML file"
    )
    parser.add_argument("--title1", type=str, required=True, help="Title for plot")
    parser.add_argument(
        "--title_diff",
        type=str,
        required=True,
        help="Title for plot showing difference between CC",
    )
    parser.add_argument(
        "--suptitle",
        type=str,
        required=True,
        help="Super title for 3-panel plot",
    )
    parser.add_argument(
        "--h", type=int, default=4000, required=False, help="Image height (pixels)"
    )
    parser.add_argument(
        "--w", type=int, default=6000, required=False, help="Image width (pixels)"
    )
    args = parser.parse_args()

    if os.path.exists(args.CC0_fn):
        cv_file = cv2.FileStorage(args.CC0_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC0_fn, cv2.FILE_STORAGE_READ)
        K0 = np.squeeze(cv_file.getNode("Camera_Matrix").mat())
        D0 = np.squeeze(cv_file.getNode("Distortion_Coefficients").mat())
        cv_file.release()
        # can't access image_Height and Width through cv reader
        tree = ET.parse(args.CC0_fn)
        h0 = int(tree.find("image_Height").text)
        w0 = int(tree.find("image_Width").text)
        tree = None
        # h0 = np.squeeze(cv_file.getNode('image_Height'))
        # w0 = np.squeeze(cv_file.getNode('image_Width'))

    if os.path.exists(args.CC1_fn):
        cv_file = cv2.FileStorage(args.CC1_fn, cv2.FILE_STORAGE_READ)
        cv_file.open(args.CC1_fn, cv2.FILE_STORAGE_READ)
        K1 = np.squeeze(cv_file.getNode("Camera_Matrix").mat())
        D1 = np.squeeze(cv_file.getNode("Distortion_Coefficients").mat())
        tree = ET.parse(args.CC1_fn)
        h1 = int(tree.find("image_Height").text)
        w1 = int(tree.find("image_Width").text)
        # h1 = np.squeeze(cv_file.getNode('image_Height'))
        # w1 = np.squeeze(cv_file.getNode('image_Width'))
        cv_file.release()

    visualizeDistortion_comparison_2image(
        args.save_fname_2panel,
        args.save_fname_2panel_diff,
        args.save_fname_1panel,
        K0,
        D0,
        K1,
        D1,
        h0,
        w0,
        fontsize=14,
        contourLevels=5,
        nstep=400,
        arrowmarkersize=2,
        title0=args.title0,
        title1=args.title1,
        title_diff=args.title_diff,
        suptitle=args.suptitle,
    )
