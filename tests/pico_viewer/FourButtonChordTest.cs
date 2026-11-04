using System;
using Vita.Camera;

public static class FourButtonChordTest
{
    private static int assertions;
    private static void Check(bool result, string message)
    {
        assertions++;
        if (!result) throw new Exception(message);
    }

    public static int Main()
    {
        var c = new FourButtonChord();
        Check(!c.Update(true, 15, 0), "Held on connect must not fire");
        Check(!c.Update(true, 15, 1), "Held on connect must remain disarmed");
        Check(!c.Update(true, 0, 2), "Neutral arms without firing");
        Check(!c.Update(true, 15, 3), "Start of chord must not fire");
        Check(!c.Update(true, 15, 3.49), "Short press must not fire");
        Check(c.Update(true, 15, 3.5), "Full hold fires exactly once");
        Check(!c.Update(true, 15, 5), "Long hold must not repeat");
        Check(!c.Update(true, 7, 6), "Partial release must not rearm");
        Check(!c.Update(true, 15, 7), "Partial release/repress must not rearm");
        Check(!c.Update(true, 15, 8), "Partial release still disarmed");
        c.Update(true, 0, 9);
        c.Update(true, 15, 10);
        Check(!c.Update(false, 15, 10.5), "Disconnect must disarm");
        Check(!c.Update(true, 15, 12), "Reconnect held must not fire");
        c.Update(true, 0, 13);
        c.Update(true, 15, 14);
        c.Update(true, 7, 14.3);
        c.Update(true, 15, 14.4);
        Check(!c.Update(true, 15, 14.8), "Broken hold must restart its timer");
        Check(c.Update(true, 15, 14.91), "Fresh full hold fires");
        c.Update(true, 0, 15);
        c.Update(true, 15, 16);
        Check(!c.Update(true, 15, 15.9), "Clock regression must disarm");
        Check(!c.Update(true, 15, 18), "Regressed clock requires release");
        c.Update(true, 0, 19);
        Check(!c.Update(true, 15, double.NaN), "Nonfinite clock must disarm");
        c.Update(true, 0, 20);
        Check(!c.Update(true, 16, 21), "Unknown button mask must disarm");
        Check(!c.Update(true, 15, 22), "Invalid mask requires release");
        c.Reset();
        Check(!c.Update(true, 15, 30), "Lifecycle reset must require release");
        Console.WriteLine("FourButtonChord: "+assertions+" assertions passed");
        return 0;
    }
}
