#!/usr/bin/env bash
# Pin Thor's clocks for speed measurements (2026-11-09, Tripp's decision: MAXN, clocks locked at the mode's maximum by
# jetson_clocks, persistent across reboots). Run once on Thor as root: sudo bash pin_power.sh   (safe to run again)
#
# The mode needs no config edit: nvpmodel saves the chosen mode itself (/var/lib/nvpmodel/status) and restores it at
# boot. (The first version also edited /etc/nvpmodel.conf, which on Thor is a symlink that sed -i would have replaced
# with a plain file; removed in the step-4 red-team before the script was ever run.) This script sets MAXN (mode 0) and
# installs a oneshot unit that runs jetson_clocks after nvpmodel.service (itself a oneshot, checked) at every boot.
# Undo: systemctl disable --now jetson-clocks.service; rm /etc/systemd/system/jetson-clocks.service;
#       systemctl daemon-reload; nvpmodel -m 1 (120W, the shipped default); reboot (unlocks the clocks).
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
[ "$(systemctl show -p Type --value nvpmodel.service)" = oneshot ] || { echo "nvpmodel.service is not a oneshot: ordering unsafe"; exit 1; }

# 1. MAXN now, if not already (nvpmodel may ask to reboot for some settings; then run this again after the reboot)
if ! nvpmodel -q | grep -q "^NV Power Mode: MAXN$"; then nvpmodel -m 0; fi
nvpmodel -q | grep -q "^NV Power Mode: MAXN$" || { echo "not in MAXN yet (a reboot may be pending): reboot, then run this again"; exit 1; }

# 2. jetson_clocks at every boot, after nvpmodel has restored the mode
cat > /etc/systemd/system/jetson-clocks.service <<'EOF'
[Unit]
Description=Lock CPU/GPU/EMC clocks at the power mode's maximum (jetson_clocks), for reproducible speed measurements
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable jetson-clocks.service
systemctl restart jetson-clocks.service   # runs jetson_clocks now, also on a second run of this script

# 3. Show the result
nvpmodel -q
jetson_clocks --show
