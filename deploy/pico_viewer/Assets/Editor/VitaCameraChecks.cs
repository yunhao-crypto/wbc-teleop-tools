using System;
using UnityEngine;

public static class VitaCameraChecks
{
    // Tests the real Unity camera hierarchy used by the pinned client. A mono
    // texture must reach both eye cameras, including when main tags the left eye.
    public static void Run()
    {
        var head = new GameObject("Vita test head");
        var left = new GameObject("Vita test left");
        var right = new GameObject("Vita test right");
        var ui = new GameObject("Vita test unrelated UI");
        try
        {
            var centerCamera = head.AddComponent<Camera>();
            left.transform.SetParent(head.transform, false);
            right.transform.SetParent(head.transform, false);
            var leftCamera = left.AddComponent<Camera>();
            var rightCamera = right.AddComponent<Camera>();
            var uiCamera = ui.AddComponent<Camera>();
            centerCamera.stereoTargetEye = StereoTargetEyeMask.Both;
            leftCamera.stereoTargetEye = StereoTargetEyeMask.Left;
            rightCamera.stereoTargetEye = StereoTargetEyeMask.Right;
            uiCamera.stereoTargetEye = StereoTargetEyeMask.None;
            Require(VitaCameraView.IsCameraInHeadRig(rightCamera, leftCamera), "Right eye missing with left as main");
            Require(VitaCameraView.IsCameraInHeadRig(centerCamera, leftCamera), "Center camera missing");
            Require(VitaCameraView.IsCameraInHeadRig(leftCamera, centerCamera), "Left eye missing with center as main");
            Require(VitaCameraView.IsCameraInHeadRig(rightCamera, centerCamera), "Right eye missing with center as main");
            Require(!VitaCameraView.IsCameraInHeadRig(uiCamera, leftCamera), "Unrelated camera was included");
            Require(!VitaCameraView.IsCameraInHeadRig(null, leftCamera), "Null camera included");
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(left);
            UnityEngine.Object.DestroyImmediate(right);
            UnityEngine.Object.DestroyImmediate(head);
            UnityEngine.Object.DestroyImmediate(ui);
        }
        Debug.Log("Vita camera Unity hierarchy checks passed (6 assertions)");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }
}
