using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using Robot;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using Vita.Camera;

// Runs beside XRoboToolkit 1.1.1. It never changes tracking or Send state.
[DefaultExecutionOrder(-1000)]
public sealed class VitaCameraView : MonoBehaviour
{
    [Serializable]
    private sealed class Config
    {
        public bool enabled = true;
        public string robot_host = "";
        public int port = 13579;
        public int width = 960;
        public int height = 768;
        public int fps = 30;
    }

    [Serializable]
    private sealed class ReceiverStatus
    {
        public string state = "";
        public string error = "";
        public string decoder = "";
        public long frames = 0;
        public float frame_age_s = -1;
    }

    private static VitaCameraView instance;
    public static bool ReserveFaceButtons => instance != null && instance.config.enabled;
    private Config config = new Config();
    private readonly FourButtonChord chord = new FourButtonChord();
    private readonly Dictionary<Canvas, bool> canvasStates = new Dictionary<Canvas, bool>();
    private readonly Dictionary<UnityEngine.Camera, int> cameraMasks = new Dictionary<UnityEngine.Camera, int>();
    private AndroidJavaObject receiver;
    private Texture2D texture;
    private GameObject screen, backing;
    private Material screenMaterial, backingMaterial;
    private UnityEngine.Camera viewCamera;
    private string host = "", activeHost = "";
    private bool requested, displayed, stopping, paused;
    private bool shortcutUiUpdated;
    private float nextStatus, nextTexturePoll, retryAfter, lastUpload = -100;
    private float nextDiagnostics;
    private ReceiverStatus status = new ReceiverStatus();
    private int videoLayer;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void Bootstrap()
    {
        if (instance != null) return;
        var root = new GameObject("Vita Camera Viewer");
        DontDestroyOnLoad(root);
        root.AddComponent<VitaCameraView>();
    }

    private void Awake()
    {
        instance = this;
        string path = Path.Combine(Application.persistentDataPath, "vita_camera.json");
        // The build can provide a private device profile as a Resources asset.
        TextAsset built = Resources.Load<TextAsset>("vita_camera");
        try
        {
            if (File.Exists(path)) config = JsonUtility.FromJson<Config>(File.ReadAllText(path));
            else if (built != null) config = JsonUtility.FromJson<Config>(built.text);
            if (config == null || config.port < 1 || config.port > 65535 ||
                config.width != 960 || config.height != 768 || config.fps != 30)
                throw new InvalidDataException("Unsupported Vita camera profile");
        }
        catch (Exception error)
        {
            config = new Config { enabled = false };
            Debug.LogError("Vita camera configuration: "+error.Message);
        }
        videoLayer = LayerMask.NameToLayer("VitaVideo");
        if (videoLayer < 0 && config.enabled)
        {
            config.enabled = false;
            Debug.LogError("VitaVideo layer is missing; use VitaCameraBuilder.BuildBatch");
        }
    }

    private static bool ReadButtons(out int buttons)
    {
        buttons = 0;
        InputDevice left = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        InputDevice right = InputDevices.GetDeviceAtXRNode(XRNode.RightHand);
        bool a, b, x, y;
        if (!left.isValid || !right.isValid ||
            !left.TryGetFeatureValue(CommonUsages.primaryButton, out x) ||
            !left.TryGetFeatureValue(CommonUsages.secondaryButton, out y) ||
            !right.TryGetFeatureValue(CommonUsages.primaryButton, out a) ||
            !right.TryGetFeatureValue(CommonUsages.secondaryButton, out b)) return false;
        buttons = (a ? 1 : 0) | (b ? 2 : 0) | (x ? 4 : 0) | (y ? 8 : 0);
        return true;
    }

    private void Update()
    {
        if (!config.enabled || paused) return;
        if (!shortcutUiUpdated)
        {
            UIOperate controls = FindObjectOfType<UIOperate>();
            if (controls != null && controls.AcontrolerTog != null)
            {
                controls.AcontrolerTog.SetIsOnWithoutNotify(false);
                controls.AcontrolerTog.interactable = false;
                shortcutUiUpdated = true;
            }
        }
        float now = Time.realtimeSinceStartup;
        int buttons;
        if (chord.Update(ReadButtons(out buttons), buttons, now))
        {
            requested = !requested;
            Debug.Log("Vita camera view requested="+requested);
            Toast.Show(requested ? "Waiting for robot camera" : "Observer view");
        }
        host = string.IsNullOrEmpty(config.robot_host) ? TcpHandler.GetTargetIP : config.robot_host;
        IPAddress address;
        bool validHost = IPAddress.TryParse(host, out address) &&
            address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork &&
            !IPAddress.IsLoopback(address) && !address.Equals(IPAddress.Any);
        if (receiver != null && (host != activeHost || stopping ||
            viewCamera == null || !viewCamera.isActiveAndEnabled))
        {
            StopReceiver();
            TryRelease();
            SetDisplayed(false);
            return;
        }
        if (receiver == null && validHost && now >= retryAfter &&
            Application.platform == RuntimePlatform.Android)
        {
            try { StartReceiver(host); }
            catch (Exception error)
            {
                Debug.LogWarning("Vita camera startup: "+error.Message);
                retryAfter = now+3;
                StopReceiver();
            }
        }
        if (receiver != null && !stopping)
        {
            try
            {
                if (now >= nextTexturePoll)
                {
                    nextTexturePoll = now+1f/60;
                    if (receiver.Call<bool>("updateTexture"))
                    {
                        GL.InvalidateState();
                        lastUpload = now;
                    }
                }
                if (now >= nextStatus)
                {
                    nextStatus = now+.1f;
                    status = JsonUtility.FromJson<ReceiverStatus>(receiver.Call<string>("status"));
                    if (now >= nextDiagnostics && status != null)
                    {
                        nextDiagnostics = now+5;
                        Debug.Log("Vita camera: state="+status.state+" decoder="+status.decoder+
                            " frames="+status.frames+" age="+status.frame_age_s+" error="+status.error);
                    }
                }
            }
            catch (Exception error)
            {
                Debug.LogWarning("Vita camera receiver: "+error.Message);
                StopReceiver();
            }
        }
        bool fresh = receiver != null && !stopping && status != null &&
            status.state == "streaming" && status.frame_age_s >= 0 && status.frame_age_s < .3f &&
            now-lastUpload < .3f;
        SetDisplayed(requested && fresh);
    }

    private void StartReceiver(string robotHost)
    {
        viewCamera = UnityEngine.Camera.main;
        if (viewCamera == null) return;
        Shader shader = Shader.Find("Unlit/Texture");
        Shader black = Shader.Find("Unlit/Color");
        if (shader == null || black == null) throw new InvalidOperationException("Video shaders unavailable");
        texture = new Texture2D(config.width, config.height, TextureFormat.RGB24, false, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        texture.Apply(false, false);
        receiver = new AndroidJavaObject("com.vita.xr.camera.VitaCameraReceiver");
        receiver.Call("initialize", texture.GetNativeTexturePtr().ToInt32(), config.width, config.height, config.fps);
        screenMaterial = new Material(shader) { mainTexture = texture };
        backingMaterial = new Material(black) { color = Color.black };
        screen = CreateQuad("Vita mono video", screenMaterial, .65f, new Vector3(1.5f, 1.2f, 1));
        backing = CreateQuad("Vita video background", backingMaterial, .66f, new Vector3(20, 20, 1));
        screen.SetActive(false); backing.SetActive(false);
        activeHost = robotHost;
        lastUpload = -100;
        status = new ReceiverStatus();
        receiver.Call("start", robotHost, config.port);
        Debug.Log("Vita camera receiver started; A+B+X+Y switches the view");
    }

    private GameObject CreateQuad(string name, Material material, float z, Vector3 size)
    {
        var obj = GameObject.CreatePrimitive(PrimitiveType.Quad);
        obj.name = name;
        Destroy(obj.GetComponent<Collider>());
        obj.layer = videoLayer;
        obj.transform.SetParent(viewCamera.transform, false);
        obj.transform.localPosition = new Vector3(0, 0, z);
        obj.transform.localRotation = Quaternion.identity;
        obj.transform.localScale = size;
        obj.GetComponent<MeshRenderer>().sharedMaterial = material;
        return obj;
    }

    private void SetDisplayed(bool on)
    {
        if (on == displayed) return;
        displayed = on;
        Debug.Log("Vita camera view displayed="+on);
        if (on)
        {
            foreach (Canvas canvas in FindObjectsOfType<Canvas>(true))
            {
                canvasStates[canvas] = canvas.enabled;
                canvas.enabled = false; // Drawing only; tracking scripts keep running.
            }
            foreach (UnityEngine.Camera camera in UnityEngine.Camera.allCameras)
            {
                cameraMasks[camera] = camera.cullingMask;
                camera.cullingMask = IsCameraInHeadRig(camera, viewCamera) ? camera.cullingMask | (1 << videoLayer)
                    : camera.cullingMask & ~(1 << videoLayer);
            }
        }
        else
        {
            foreach (var saved in canvasStates) if (saved.Key != null) saved.Key.enabled = saved.Value;
            canvasStates.Clear();
            foreach (var saved in cameraMasks) if (saved.Key != null) saved.Key.cullingMask = saved.Value;
            cameraMasks.Clear();
            if (requested && !paused) Toast.Show("Camera unavailable; observer view");
        }
        if (screen != null) screen.SetActive(on);
        if (backing != null) backing.SetActive(on);
    }

    // XRoboToolkit 1.1.1 uses separate Left/Right cameras under a tracked
    // center camera. Camera.main can be the left eye or the center camera.
    public static bool IsCameraInHeadRig(UnityEngine.Camera candidate, UnityEngine.Camera main)
    {
        if (candidate == null || main == null) return false;
        if (candidate == main) return true;
        if (candidate.stereoTargetEye == StereoTargetEyeMask.None) return false;
        Transform root = main.transform.parent;
        return candidate.transform.parent == main.transform ||
            (root != null && (candidate.transform == root || candidate.transform.parent == root));
    }

    private void StopReceiver()
    {
        if (receiver == null || stopping) return;
        stopping = true;
        receiver.Call("stop");
    }

    private void TryRelease()
    {
        if (receiver == null || !receiver.Call<bool>("releaseIfStopped")) return;
        receiver.Dispose(); receiver = null; stopping = false;
        if (screen != null) Destroy(screen);
        if (backing != null) Destroy(backing);
        if (screenMaterial != null) Destroy(screenMaterial);
        if (backingMaterial != null) Destroy(backingMaterial);
        if (texture != null) Destroy(texture);
        screen = backing = null;
        texture = null;
        retryAfter = Time.realtimeSinceStartup+1;
    }

    private void OnApplicationPause(bool pause)
    {
        paused = pause;
        chord.Reset();
        if (pause) { SetDisplayed(false); StopReceiver(); }
    }

    private void OnDestroy()
    {
        SetDisplayed(false);
        StopReceiver();
        TryRelease();
        if (instance == this) instance = null;
    }
}
