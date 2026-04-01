setTimeout(() => {
  Java.perform(() => {
    console.log("Minimal spoofing test...");

    const Build = Java.use("android.os.Build");
    Build.MODEL.value = "Pixel 9999a";
    Build.MANUFACTURER.value = "FakeCo";

    const Build_VERSION = Java.use("android.os.Build$VERSION");
    Build_VERSION.SDK_INT.implementation = function () {
      return 65535;
    };
    Build_VERSION.RELEASE.implementation = function () {
      return "35";
    };

    console.log("Spoofed Build info safely.");
  });
}, 100);
