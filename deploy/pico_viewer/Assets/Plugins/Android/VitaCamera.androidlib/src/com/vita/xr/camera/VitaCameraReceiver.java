package com.vita.xr.camera;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;
import android.view.Surface;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** HEVC client. Network/codec work never runs on Unity's tracking thread. */
public final class VitaCameraReceiver {
    private Object fbo;
    private Method hasFrameMethod, updateMethod, releaseMethod;
    private Surface surface;
    private int width, height;
    private volatile boolean running;
    private volatile Socket socket;
    private Thread worker;
    private volatile String state = "idle", error = "";
    private volatile long lastOutputNs, outputs;
    private MediaCodec codec; // Owned only by the network worker.
    private final MediaCodec.BufferInfo outputInfo = new MediaCodec.BufferInfo();
    private String decoderName;

    /** Called on Unity's GL thread with the existing XRoboToolkit FBO library. */
    public void initialize(int textureId, int width, int height, int fps) throws Exception {
        this.width = width; this.height = height;
        MediaCodecInfo chosen = null;
        for (MediaCodecInfo info : new MediaCodecList(MediaCodecList.ALL_CODECS).getCodecInfos()) {
            if (info.isEncoder() || !info.isHardwareAccelerated()) continue;
            for (String type : info.getSupportedTypes()) {
                if (!MediaFormat.MIMETYPE_VIDEO_HEVC.equalsIgnoreCase(type)) continue;
                MediaCodecInfo.CodecCapabilities caps = info.getCapabilitiesForType(type);
                if (caps.getVideoCapabilities().areSizeAndRateSupported(width, height, fps)) {
                    chosen = info; break;
                }
            }
            if (chosen != null) break;
        }
        if (chosen == null) throw new IOException("No hardware HEVC decoder for this video profile");
        decoderName = chosen.getName();
        // Reflection lets this small Android library reuse the shipped FBO AAR
        // without copying it or adding a dependency on the enterprise SDK.
        Class<?> type = Class.forName("com.picoxr.fbolibrary.FBOPlugin");
        fbo = type.getDeclaredConstructor().newInstance();
        hasFrameMethod = type.getMethod("isUpdateFrame");
        updateMethod = type.getMethod("updateTexture");
        releaseMethod = type.getMethod("release");
        type.getMethod("init").invoke(fbo);
        type.getMethod("BuildTexture", int.class, int.class, int.class).invoke(fbo, textureId, width, height);
        surface = (Surface) type.getMethod("getSurface").invoke(fbo);
        state = "initialized";
    }

    public synchronized void start(String host, int port) {
        if (running) return;
        running = true;
        worker = new Thread(() -> receive(host, port), "vita-camera-hevc");
        worker.setDaemon(true);
        worker.start();
    }

    private void createCodec() throws IOException {
        releaseCodec();
        MediaFormat format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_HEVC, width, height);
        format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, VideoWire.MAX_PAYLOAD);
        format.setInteger(MediaFormat.KEY_PRIORITY, 0);
        format.setInteger("low-latency", 1);
        codec = MediaCodec.createByCodecName(decoderName);
        codec.configure(format, surface, null, 0);
        codec.start();
    }

    private void releaseCodec() {
        MediaCodec old = codec;
        codec = null;
        if (old != null) {
            try { old.stop(); } catch (Exception ignored) {}
            try { old.release(); } catch (Exception ignored) {}
        }
    }

    private void receive(String host, int port) {
        while (running) {
            state = "connecting";
            try (Socket connection = new Socket()) {
                socket = connection;
                connection.connect(new InetSocketAddress(host, port), 2000);
                connection.setSoTimeout(1500);
                connection.setTcpNoDelay(true);
                connection.setReceiveBufferSize(64*1024);
                DataInputStream input = new DataInputStream(new BufferedInputStream(connection.getInputStream(), 32768));
                VideoWire.Packet hello = VideoWire.read(input);
                if (hello.kind != VideoWire.CONFIG) throw new IOException("Missing camera configuration");
                JSONObject config = new JSONObject(new String(hello.payload, StandardCharsets.UTF_8));
                if (!"vita_pico_camera_v1".equals(config.getString("schema")) ||
                    !"hevc".equals(config.getString("codec")) ||
                    !"mono".equals(config.getString("stereo_layout")) ||
                    config.getInt("width") != width || config.getInt("height") != height)
                    throw new IOException("Camera configuration differs from the viewer");
                createCodec();
                boolean waitingKeyframe = true;
                long previousSequence = -1, originStamp = 0, previousPts = -1;
                state = "waiting_keyframe";
                error = "";
                while (running) {
                    VideoWire.Packet frame = VideoWire.read(input);
                    if (frame.kind == VideoWire.STATUS) continue;
                    if (frame.kind != VideoWire.VIDEO) throw new IOException("Unexpected camera packet");
                    boolean key = (frame.flags & VideoWire.KEYFRAME) != 0;
                    boolean reset = (frame.flags & VideoWire.DISCONTINUITY) != 0 ||
                        (previousSequence >= 0 && frame.sequence != previousSequence+1);
                    previousSequence = frame.sequence;
                    if (reset) {
                        createCodec();
                        waitingKeyframe = true;
                        originStamp = 0; previousPts = -1;
                    }
                    if (waitingKeyframe && !key) continue;
                    if (originStamp == 0) originStamp = frame.stampNs;
                    long pts = Math.max(previousPts+1, (frame.stampNs-originStamp)/1000);
                    if (!submit(frame.payload, pts)) {
                        // A skipped reference frame invalidates the GOP. Resume
                        // only at an independent keyframe, never the next P frame.
                        createCodec();
                        waitingKeyframe = true;
                        state = "waiting_keyframe";
                        originStamp = 0; previousPts = -1;
                        continue;
                    }
                    waitingKeyframe = false;
                    previousPts = pts;
                    state = "streaming";
                }
            } catch (Exception exception) {
                error = exception.getClass().getSimpleName()+": "+exception.getMessage();
                state = running ? "reconnecting" : "stopped";
            } finally {
                socket = null;
                lastOutputNs = 0;
                releaseCodec();
            }
            if (running) {
                try { Thread.sleep(500); } catch (InterruptedException ignored) {}
            }
        }
        state = "stopped";
    }

    private boolean submit(byte[] payload, long pts) {
        long deadline = System.nanoTime()+100_000_000L;
        do {
            drain();
            int slot = codec.dequeueInputBuffer(2000);
            if (slot >= 0) {
                ByteBuffer buffer = codec.getInputBuffer(slot);
                if (buffer == null || buffer.capacity() < payload.length) return false;
                buffer.clear(); buffer.put(payload);
                codec.queueInputBuffer(slot, 0, payload.length, pts, 0);
                drain();
                return true;
            }
        } while (running && System.nanoTime() < deadline);
        return false;
    }

    private void drain() {
        for (int count=0; count<16; count++) {
            int slot = codec.dequeueOutputBuffer(outputInfo, 1000);
            if (slot >= 0) {
                codec.releaseOutputBuffer(slot, true);
                lastOutputNs = System.nanoTime();
                outputs++;
            } else if (slot != MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) break;
        }
    }

    public boolean updateTexture() throws Exception {
        if (fbo != null && (Boolean) hasFrameMethod.invoke(fbo)) {
            updateMethod.invoke(fbo);
            return true;
        }
        return false;
    }

    public String status() throws Exception {
        JSONObject result = new JSONObject();
        result.put("state", state); result.put("error", error);
        result.put("decoder", decoderName); result.put("frames", outputs);
        long last = lastOutputNs;
        result.put("frame_age_s", last == 0 ? -1. : (System.nanoTime()-last)*1e-9);
        return result.toString();
    }

    /** Stop sockets immediately; only the worker owns codec shutdown. */
    public synchronized void stop() {
        running = false;
        Socket current = socket;
        if (current != null) try { current.close(); } catch (IOException ignored) {}
        if (worker != null) worker.interrupt();
    }

    /** Poll on Unity's GL thread; never block its pose transmission loop. */
    public boolean releaseIfStopped() throws Exception {
        if (worker != null && worker.isAlive()) return false;
        if (fbo != null) {
            releaseMethod.invoke(fbo);
            fbo = null; surface = null;
        }
        return true;
    }
}
