// spoof.js — ICSE demo hardcoded spoofing
setTimeout(() => {
    Java.perform(() => {
        console.log("Frida script started.");

        // ——— Context / services ———
        const ActivityThread = Java.use("android.app.ActivityThread");
        const currentApplication = ActivityThread.currentApplication();
        const context = currentApplication.getApplicationContext();

        const SensorManager = Java.use("android.hardware.SensorManager");
        const sensorManager = Java.cast(context.getSystemService("sensor"), SensorManager);

        const Sensor = Java.use("android.hardware.Sensor");
        const SENSOR_TYPE_ACCELEROMETER = Sensor.TYPE_ACCELEROMETER.value;
        const SENSOR_TYPE_GYROSCOPE = Sensor.TYPE_GYROSCOPE.value;
        const SENSOR_TYPE_LINEAR_ACCELERATION = Sensor.TYPE_LINEAR_ACCELERATION.value;
        const SENSOR_TYPE_ORIENTATION = Sensor.TYPE_ORIENTATION.value;
        const SENSOR_TYPE_LIGHT = Sensor.TYPE_LIGHT.value;
        const SENSOR_TYPE_PRESSURE = Sensor.TYPE_PRESSURE.value;
        const SENSOR_TYPE_PROXIMITY = Sensor.TYPE_PROXIMITY.value;
        const SENSOR_TYPE_GRAVITY = Sensor.TYPE_GRAVITY.value;
        const SENSOR_TYPE_ROTATION_VECTOR = Sensor.TYPE_ROTATION_VECTOR.value;
        const SENSOR_TYPE_STEP_COUNTER = Sensor.TYPE_STEP_COUNTER.value;
        const SENSOR_TYPE_MAGNETIC_FIELD = Sensor.TYPE_MAGNETIC_FIELD.value;

        const accelerometerSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_ACCELEROMETER);
        const gyroscopeSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_GYROSCOPE);
        const linearAccelerometerSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_LINEAR_ACCELERATION);
        const orientationSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_ORIENTATION);
        const lightSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_LIGHT);
        const pressureSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_PRESSURE);
        const proximitySensor = sensorManager.getDefaultSensor(SENSOR_TYPE_PROXIMITY);
        const gravitySensor = sensorManager.getDefaultSensor(SENSOR_TYPE_GRAVITY);
        const rotationVectorSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_ROTATION_VECTOR);
        const stepCounterSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_STEP_COUNTER);
        const geomagneticFieldSensor = sensorManager.getDefaultSensor(SENSOR_TYPE_MAGNETIC_FIELD);

        const SystemSensorManager = Java.use("android.hardware.SystemSensorManager");
        const SensorEventQueue = Java.use("android.hardware.SystemSensorManager$SensorEventQueue");

        // Save original to avoid recursion
        const _dispatch = SensorEventQueue.dispatchSensorEvent;

        // ——— Battery spoof: BatteryManager API + sticky intent ———
        const BatteryManager = Java.use("android.os.BatteryManager");
        const _BM_getIntProperty = BatteryManager.getIntProperty.overload('int');
        BatteryManager.getIntProperty.overload('int').implementation = function (id) {
            if (id === BatteryManager.BATTERY_PROPERTY_CAPACITY.value) return 5;
            return _BM_getIntProperty.call(this, id);
        };
        const _BM_isCharging = BatteryManager.isCharging.overload();
        BatteryManager.isCharging.overload().implementation = function () { return true; };

        const Intent = Java.use('android.content.Intent');
        const Context = Java.use('android.content.Context');
        const IntentFilter = Java.use('android.content.IntentFilter');
        const _registerReceiver = Context.registerReceiver.overload('android.content.BroadcastReceiver', 'android.content.IntentFilter');

        Context.registerReceiver.overload('android.content.BroadcastReceiver', 'android.content.IntentFilter').implementation = function (rcv, filter) {
            const intent = _registerReceiver.call(this, rcv, filter);
            try {
                if (filter && filter.actionsIterator && filter.actionsIterator().hasNext()) {
                    const it = filter.actionsIterator();
                    const first = it.next();
                    if (first === 'android.intent.action.BATTERY_CHANGED') {
                        const fake = Intent.$new('android.intent.action.BATTERY_CHANGED');
                        fake.putExtra('level', 5);
                        fake.putExtra('scale', 100);
                        fake.putExtra('status', 2);   // CHARGING
                        fake.putExtra('plugged', 1);  // AC
                        fake.putExtra('health', 2);   // GOOD
                        fake.putExtra('present', true);
                        fake.putExtra('temperature', 5);
                        fake.putExtra('voltage', 5);
                        fake.putExtra('technology', 'Li-🤡');
                        return fake;
                    }
                }
            } catch (_) { }
            return intent;
        };

        // Battery extras even when they hold on to the Intent
        const _getIntExtra = Intent.getIntExtra.overload('java.lang.String', 'int');
        const _getStringExtra = Intent.getStringExtra.overload('java.lang.String');

        function isBatteryIntent(self) {
            try { return String(self.getAction()) === 'android.intent.action.BATTERY_CHANGED'; }
            catch (_) { return false; }
        }
        Intent.getIntExtra.overload('java.lang.String', 'int').implementation = function (k, d) {
            if (isBatteryIntent(this)) {
                if (k === 'level') return 5;
                if (k === 'scale') return 100;
                if (k === 'status') return 2;      // CHARGING
                if (k === 'plugged') return 1;     // AC
                if (k === 'health') return 2;      // GOOD
                if (k === 'temperature') return 5; // 0.5°C units
                if (k === 'voltage') return 5;     // 5 mV
            }
            return _getIntExtra.call(this, k, d);
        };
        Intent.getStringExtra.overload('java.lang.String').implementation = function (k) {
            if (isBatteryIntent(this) && k === 'technology') return 'Li-🤡';
            return _getStringExtra.call(this, k);
        };


        // ——— NFC spoof ———
        const NfcAdapter = Java.use('android.nfc.NfcAdapter');
        const _NA_getDefault = NfcAdapter.getDefaultAdapter.overload('android.content.Context');
        const _NA_isEnabled = NfcAdapter.isEnabled.overload();
        NfcAdapter.getDefaultAdapter.overload('android.content.Context').implementation = function (ctx) {
            return _NA_getDefault.call(this, ctx);
        };
        NfcAdapter.isEnabled.overload().implementation = function () { return false; };

        // ——— Sensor spoof ———
        SensorEventQueue.dispatchSensorEvent.implementation = function (handle, values, accuracy, timestamp) {
            try {
                if (accelerometerSensor && handle === accelerometerSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Accelerometer] ${values}`);
                }
                if (gyroscopeSensor && handle === gyroscopeSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Gyroscope] ${values}`);
                }
                if (linearAccelerometerSensor && handle === linearAccelerometerSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Linear Accel] ${values}`);
                }
                if (geomagneticFieldSensor && handle === geomagneticFieldSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Mag Field] ${values}`);
                }
                if (lightSensor && handle === lightSensor.getHandle()) {
                    values[0] = 500;
                    console.log(`[Light] ${values[0]} lx`);
                }
                if (pressureSensor && handle === pressureSensor.getHandle()) {
                    values[0] = 5;
                    console.log(`[Pressure] ${values[0]} hPa`);
                }
                if (proximitySensor && handle === proximitySensor.getHandle()) {
                    values[0] = 5;
                    console.log(`[Proximity] ${values[0]} cm`);
                }
                if (stepCounterSensor /* && handle === stepCounterSensor.getHandle() */) {
                    values[0] = 5;
                    console.log(`[Step Counter] ${values[0]}`);
                }
                if (gravitySensor && handle === gravitySensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Gravity] ${values}`);
                }
                if (rotationVectorSensor && handle === rotationVectorSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Rotation] ${values}`);
                }
                if (orientationSensor && handle === orientationSensor.getHandle()) {
                    values[0] = 5; values[1] = 5; values[2] = 5;
                    console.log(`[Orientation] ${values}`);
                }
            } catch (e) {
                console.log("Sensor spoof error: " + e);
            }
            return _dispatch.call(this, handle, values, accuracy, timestamp);
        };

        // ——— Build / System properties spoof ———
        const Build = Java.use("android.os.Build");
        const Build_VERSION = Java.use('android.os.Build$VERSION');
        Build.MODEL.value = "Pixel 9999a ??? 🦄";
        Build.MANUFACTURER.value = "RANDOM COMPANY";
        Build.BRAND.value = "42";
        Build.PRODUCT.value = "time_machine";
        Build.DEVICE.value = "toaster";
        Build_VERSION.RELEASE.value = "35";



        Build_VERSION.SDK_INT.value = 65535;

        const SystemProperties = Java.use("android.os.SystemProperties");
        const _SP_get = SystemProperties.get.overload('java.lang.String');
        const _SP_get_def = SystemProperties.get.overload('java.lang.String', 'java.lang.String');
        const _SP_getInt = SystemProperties.getInt.overload('java.lang.String', 'int');
        const _SP_getLong = SystemProperties.getLong.overload('java.lang.String', 'long');

        function spoofProp(key, orig) {
            if (key === "ro.product.model") return "Pixel 9999a ??? 🦄";
            if (key === "ro.product.manufacturer") return "RANDOM COMPANY";
            if (key === "ro.build.version.release") return "35";
            if (key.indexOf("version.sdk") >= 0 || key === "ro.product.first_api_level") return "65535";
            return orig(key);
        }
        SystemProperties.get.overload('java.lang.String').implementation = function (k) {
            return spoofProp(k, (x) => _SP_get.call(this, x));
        };
        SystemProperties.get.overload('java.lang.String', 'java.lang.String').implementation = function (k, d) {
            const v = spoofProp(k, (x) => _SP_get.call(this, x));
            return (v == null || v.length === 0) ? d : v;
        };
        SystemProperties.getInt.implementation = function (k, d) {
            if (k.indexOf("version.sdk") >= 0) return 65535;
            return _SP_getInt.call(this, k, d);
        };
        SystemProperties.getLong.implementation = function (k, d) {
            return _SP_getLong.call(this, k, d);
        };

        // Extra: cores / android_id
        const Runtime = Java.use("java.lang.Runtime");
        const _avail = Runtime.availableProcessors.overload();
        Runtime.availableProcessors.overload().implementation = function () { return 42; };

        const SettingsSecure = Java.use('android.provider.Settings$Secure');
        const _SS_getString = SettingsSecure.getString.overload('android.content.ContentResolver', 'java.lang.String');
        SettingsSecure.getString.overload('android.content.ContentResolver', 'java.lang.String').implementation = function (resolver, key) {
            if (key === "android_id") return "deadbeefdeadbeef";
            return _SS_getString.call(this, resolver, key);
        };

        console.log("Frida script setup complete.");
    });
}, 100);
