using System;

namespace Vita.Camera
{
    // Independent of Unity so the actual gesture state machine runs in host tests.
    public sealed class FourButtonChord
    {
        private readonly double holdSeconds;
        private bool armed;
        private double heldSince = double.NaN;
        private double previous = double.NegativeInfinity;

        public FourButtonChord(double holdSeconds = 0.5)
        {
            if (double.IsNaN(holdSeconds) || double.IsInfinity(holdSeconds) || holdSeconds <= 0)
                throw new ArgumentOutOfRangeException(nameof(holdSeconds));
            this.holdSeconds = holdSeconds;
        }

        public void Reset()
        {
            armed = false;
            heldSince = double.NaN;
            previous = double.NegativeInfinity;
        }

        // Bits: A=1, B=2, X=4, Y=8. Both controllers must have fresh readings.
        public bool Update(bool connected, int buttons, double now)
        {
            if (!connected || buttons < 0 || buttons > 15 || double.IsNaN(now) ||
                double.IsInfinity(now) || now < previous)
            {
                Reset();
                return false;
            }
            previous = now;
            if (buttons == 0)
            {
                armed = true;
                heldSince = double.NaN;
            }
            else if (buttons != 15)
                heldSince = double.NaN;
            else if (armed)
            {
                if (double.IsNaN(heldSince)) heldSince = now;
                if (now-heldSince >= holdSeconds)
                {
                    armed = false;
                    heldSince = double.NaN;
                    return true;
                }
            }
            return false;
        }
    }
}
