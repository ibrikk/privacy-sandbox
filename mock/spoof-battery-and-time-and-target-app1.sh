echo "[+] Spoofing battery and time..."

# Battery spoofing
su -c "dumpsys battery set level 5"
su -c "dumpsys battery set status 2"
su -c "dumpsys battery set plugged 1"
su -c "dumpsys battery set temperature 350"
su -c "dumpsys battery set health 2"
su -c "dumpsys battery set present false"

# Time spoofing
su -c "date -s 20240331.122000"  # Time spoofing (example: March 31, 2024, 12:20:00 PM)

# Timezone spoofing
su -c "setprop persist.sys.timezone Asia/Tokyo"

echo "[+] Launching Frida spoof on com.exatools.sensors..."

# Run Frida hook with spoof.js
frida -H 127.0.0.1 -f com.exatools.sensors -l /sdcard/Download/spoof.js --debug

echo "[+] All spoofing steps complete."
