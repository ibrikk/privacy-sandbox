echo "[+] Resetting battery spoof and restoring time..."


# Reset battery
su -c "dumpsys battery reset"

su -c "date -s '2025-09-16 09:03:00'"

# Enable automatic time
su -c "settings put global auto_time 1"
su -c "settings put global auto_time_zone 1"

# Trigger NTP sync via network toggle (may not work on all ROMs)
echo "[+] Toggling airplane mode to trigger time sync..."
su -c "settings put global airplane_mode_on 1"
su -c "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state true"
sleep 2
su -c "settings put global airplane_mode_on 0"
su -c "am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false"

echo "[+] Done. Battery reset and NTP sync triggered."


# Use in Terminal
# adb shell su -c "dumpsys battery reset"
# adb shell su -c "settings put global auto_time 1"
# adb shell su -c "settings put global auto_time_zone 1"
# echo "[+] Battery spoofing disabled and system time set to automatic."
