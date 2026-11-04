using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Android;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;

public static class VitaCameraBuilder
{
    public static void BuildBatch()
    {
        string sdk = Environment.GetEnvironmentVariable("VITA_ANDROID_SDK");
        string ndk = Environment.GetEnvironmentVariable("VITA_ANDROID_NDK");
        string jdk = Environment.GetEnvironmentVariable("VITA_ANDROID_JDK");
        if (!string.IsNullOrEmpty(sdk)) AndroidExternalToolsSettings.sdkRootPath = sdk;
        if (!string.IsNullOrEmpty(ndk)) AndroidExternalToolsSettings.ndkRootPath = ndk;
        if (!string.IsNullOrEmpty(jdk)) AndroidExternalToolsSettings.jdkRootPath = jdk;
        EnsureVideoLayer();
        EnsureShaders();
        VitaCameraChecks.Run();
        PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, "com.xrobotoolkit.client.vitacamera");
        PlayerSettings.productName = "XRT Vita Camera";
        PlayerSettings.bundleVersion = "1.1.1-vita.2";
        PlayerSettings.Android.bundleVersionCode = 2;
        PlayerSettings.Android.useCustomKeystore = false;
        PlayerSettings.Android.minifyRelease = false;
        PlayerSettings.Android.minifyDebug = false;
        PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel30;
        // This editor exposes only selected named API levels; numeric API 31 is valid.
        PlayerSettings.Android.targetSdkVersion = (AndroidSdkVersions)31;
        PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
        PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
        // The shipped FBO renderer uses OpenGL textures and must not run under Vulkan.
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.OpenGLES3 });
        PlayerSettings.stripEngineCode = false;
        AssetDatabase.SaveAssets();
        string path = Environment.GetEnvironmentVariable("VITA_CAMERA_APK") ??
            Path.GetFullPath("Builds/Android/XRT-Vita-Camera-1.1.1-vita.2.apk");
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions {
            scenes = EditorBuildSettings.scenes.Where(s => s.enabled).Select(s => s.path).ToArray(),
            target = BuildTarget.Android,
            locationPathName = path,
            options = BuildOptions.None
        });
        if (report.summary.result != BuildResult.Succeeded)
            throw new Exception("Vita camera APK build failed: "+report.summary.result);
        Debug.Log("VITA_CAMERA_APK="+path);
    }

    private static void EnsureVideoLayer()
    {
        var asset = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
        var layers = asset.FindProperty("layers");
        int empty = -1;
        for (int i=8; i<32; i++)
        {
            string name = layers.GetArrayElementAtIndex(i).stringValue;
            if (name == "VitaVideo") return;
            if (name == "" && empty < 0) empty = i;
        }
        if (empty < 0) throw new Exception("No free layer for Vita video");
        layers.GetArrayElementAtIndex(empty).stringValue = "VitaVideo";
        asset.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void EnsureShaders()
    {
        var settings = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/GraphicsSettings.asset")[0]);
        var shaders = settings.FindProperty("m_AlwaysIncludedShaders");
        foreach (string name in new[] { "Unlit/Texture", "Unlit/Color" })
        {
            Shader shader = Shader.Find(name);
            if (shader == null) throw new Exception("Missing shader: "+name);
            bool found = false;
            for (int i=0; i<shaders.arraySize; i++)
                if (shaders.GetArrayElementAtIndex(i).objectReferenceValue == shader) found = true;
            if (!found)
            {
                shaders.arraySize++;
                shaders.GetArrayElementAtIndex(shaders.arraySize-1).objectReferenceValue = shader;
            }
        }
        settings.ApplyModifiedPropertiesWithoutUndo();
    }
}
