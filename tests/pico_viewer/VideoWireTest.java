import com.vita.xr.camera.VideoWire;
import java.io.*;

public final class VideoWireTest {
    private static int checks;
    private static void check(boolean value) {
        checks++;
        if (!value) throw new AssertionError();
    }
    private static byte[] encode(int kind, int flags, int length, byte[] body) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bytes);
        out.writeInt(0x56544331); out.writeByte(1); out.writeByte(kind);
        out.writeShort(flags); out.writeInt(length);
        out.writeLong(1234); out.writeLong(1800000000012345678L); out.write(body);
        return bytes.toByteArray();
    }
    public static void main(String[] args) throws Exception {
        byte[] wire = encode(2, 3, 3, new byte[]{1,2,3});
        // Fragmented reads exercise readFully rather than one-read assumptions.
        InputStream fragments = new ByteArrayInputStream(wire) {
            @Override public synchronized int read(byte[] b, int off, int len) {
                return super.read(b, off, Math.min(1,len));
            }
        };
        VideoWire.Packet packet = VideoWire.read(new DataInputStream(fragments));
        check(packet.kind == 2 && packet.flags == 3);
        check(packet.sequence == 1234 && packet.stampNs == 1800000000012345678L);
        check(packet.payload.length == 3 && packet.payload[2] == 3);
        for (int length : new int[]{-1,0,VideoWire.MAX_PAYLOAD+1}) {
            try {
                VideoWire.read(new DataInputStream(new ByteArrayInputStream(encode(2,0,length,new byte[0]))));
                throw new AssertionError("Invalid length accepted");
            } catch (IOException expected) { checks++; }
        }
        try {
            VideoWire.read(new DataInputStream(new ByteArrayInputStream(encode(2,0,5,new byte[2]))));
            throw new AssertionError("Truncated packet accepted");
        } catch (EOFException expected) { checks++; }
        byte[] bad = wire.clone(); bad[0] = 0;
        try {
            VideoWire.read(new DataInputStream(new ByteArrayInputStream(bad)));
            throw new AssertionError("Invalid magic accepted");
        } catch (IOException expected) { checks++; }
        System.out.println("VideoWire: "+checks+" assertions passed");
    }
}
