package com.vita.xr.camera;

import java.io.DataInputStream;
import java.io.IOException;

/** VTC1 wire contract shared with the read-only Aorta relay. */
public final class VideoWire {
    public static final int CONFIG = 1, VIDEO = 2, STATUS = 3;
    public static final int KEYFRAME = 1, DISCONTINUITY = 2;
    public static final int MAX_PAYLOAD = 2 * 1024 * 1024;

    public static final class Packet {
        public final int kind, flags;
        public final long sequence, stampNs;
        public final byte[] payload;
        Packet(int kind, int flags, long sequence, long stampNs, byte[] payload) {
            this.kind = kind; this.flags = flags;
            this.sequence = sequence; this.stampNs = stampNs; this.payload = payload;
        }
    }

    public static Packet read(DataInputStream input) throws IOException {
        if (input.readInt() != 0x56544331 || input.readUnsignedByte() != 1)
            throw new IOException("Unsupported VTC1 header");
        int kind = input.readUnsignedByte(), flags = input.readUnsignedShort();
        int length = input.readInt();
        long sequence = input.readLong(), stamp = input.readLong();
        if (kind < CONFIG || kind > STATUS || (flags & ~3) != 0 ||
            length < 1 || length > MAX_PAYLOAD || (kind != VIDEO && length > 16384))
            throw new IOException("Invalid VTC1 packet bounds");
        byte[] payload = new byte[length];
        input.readFully(payload);
        return new Packet(kind, flags, sequence, stamp, payload);
    }

    private VideoWire() {}
}
