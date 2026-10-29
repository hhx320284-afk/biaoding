#!/usr/bin/env bash
# 作用：把右臂的 USB-CAN 适配器固定命名为 can1 并设为 1 Mbps。
#       本次立即生效；以后开机、换 USB 口重新插拔也会自动恢复成 can1。
# 做法：udev 规则按 USB 序列号认准这个适配器（不认口位），触发一次性服务执行改名和设速。
# 用法：sudo bash "/home/w/VLA/TCP标定/程序/运行支持/配置右臂CAN接口.sh"
set -euo pipefail

CAN_NAME="can1"
BITRATE="1000000"
IP_BIN="/usr/sbin/ip"
HELPER="/usr/local/sbin/can1-right-up.sh"
SERVICE_UNIT="/etc/systemd/system/can1-right-1m.service"
UDEV_RULE="/etc/udev/rules.d/90-can1-right.rules"

if [[ ${EUID} -ne 0 ]]; then
    printf '需要管理员权限。请这样运行：\n  sudo bash "%s"\n' "$0" >&2
    exit 1
fi

sysfs_serial() {  # $1 = 网卡名；沿 sysfs 向上读 USB 设备的序列号
    local path
    path="$(readlink -f "/sys/class/net/$1/device" 2>/dev/null)" || return 1
    cat "$path/../serial" 2>/dev/null
}

echo "=== 1/5 查找 USB-CAN 适配器 ==="
found=()
for dir in /sys/class/net/*; do
    name="$(basename "$dir")"
    [[ "$name" == can* ]] || continue
    [[ "$(cat "$dir/type" 2>/dev/null || echo 0)" == "280" ]] || continue
    found+=("$name")
done

if [[ ${#found[@]} -eq 0 ]]; then
    echo "没有找到 CAN 适配器。请插好 USB-CAN 后重跑本脚本。" >&2
    exit 1
fi
if [[ ${#found[@]} -gt 1 ]]; then
    echo "同时发现多个 CAN 接口：${found[*]}" >&2
    echo "为避免连错机械臂，请只保留右臂的 USB-CAN 后重跑。" >&2
    exit 1
fi

IFACE="${found[0]}"
SERIAL="$(sysfs_serial "$IFACE" || true)"
if [[ -z "$SERIAL" ]]; then
    echo "读不到 $IFACE 的 USB 序列号，无法固定绑定；请确认它是 USB-CAN 适配器。" >&2
    exit 1
fi
echo "    接口 $IFACE，USB 序列号 $SERIAL"

echo "=== 2/5 写入自动配置脚本 ${HELPER} ==="
cat > "$HELPER" <<EOF
#!/bin/sh
# 由 udev 触发：把序列号 ${SERIAL} 的 USB-CAN 固定成 ${CAN_NAME}（${BITRATE} bps）并启用。
# 由 配置右臂CAN接口.sh 生成，请勿手改。
SERIAL='${SERIAL}'
NAME='${CAN_NAME}'
IP=${IP_BIN}

i=0
iface=''
while [ \$i -lt 25 ]; do
    for d in /sys/class/net/can*; do
        [ -e "\$d" ] || continue
        p=\$(readlink -f "\$d/device" 2>/dev/null) || continue
        s=\$(cat "\$p/../serial" 2>/dev/null || true)
        [ "\$s" = "\$SERIAL" ] || continue
        iface=\$(basename "\$d")
        break
    done
    [ -n "\$iface" ] && break
    i=\$((i + 1))
    sleep 0.2
done
[ -n "\$iface" ] || exit 0

\$IP link set "\$iface" down 2>/dev/null || true
\$IP link set "\$iface" type can bitrate ${BITRATE} 2>/dev/null || true
[ "\$iface" = "\$NAME" ] || \$IP link set "\$iface" name "\$NAME" 2>/dev/null || true
\$IP link set "\$NAME" up 2>/dev/null || true
EOF
chmod 755 "$HELPER"

echo "=== 3/5 写入服务 ${SERVICE_UNIT} ==="
cat > "$SERVICE_UNIT" <<EOF
[Unit]
Description=右臂 CAN 接口 ${CAN_NAME} 设为 ${BITRATE} bps 并启用
After=systemd-udevd.service

[Service]
Type=oneshot
ExecStart=${HELPER}

[Install]
WantedBy=multi-user.target
EOF

echo "=== 4/5 写入 udev 规则 ${UDEV_RULE} ==="
cat > "$UDEV_RULE" <<'EOF'
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", TAG+="systemd", ENV{SYSTEMD_WANTS}+="can1-right-1m.service"
EOF

systemctl daemon-reload
systemctl enable can1-right-1m.service >/dev/null 2>&1 || true
udevadm control --reload-rules

echo "=== 5/5 立即应用 ==="
if [[ "$IFACE" != "$CAN_NAME" ]] && [[ -e "/sys/class/net/$CAN_NAME" ]]; then
    echo "$CAN_NAME 已被其它设备占用，请先拔掉那个适配器。" >&2
    exit 1
fi
"$IP_BIN" link set "$IFACE" down || true
"$IP_BIN" link set "$IFACE" type can bitrate "$BITRATE"
[[ "$IFACE" == "$CAN_NAME" ]] || "$IP_BIN" link set "$IFACE" name "$CAN_NAME"
"$IP_BIN" link set "$CAN_NAME" up
"$IP_BIN" -details link show "$CAN_NAME"

cat <<'EOF'

完成。回到标定界面点「连接右臂」，不用重启程序。
以后开机、换 USB 口重新插拔，都会按序列号自动恢复成 can1 + 1 Mbps。

验证方法：
  ip -details link show can1        # 应看到 UP 且 can state ERROR-ACTIVE
  sudo udevadm trigger --action=add --subsystem-match=net   # 强制重跑一次自动配置

回滚（恢复系统默认命名并删除自动配置）：
  sudo rm -f /usr/local/sbin/can1-right-up.sh \
             /etc/systemd/system/can1-right-1m.service \
             /etc/udev/rules.d/90-can1-right.rules
  sudo systemctl disable can1-right-1m.service
  sudo systemctl daemon-reload && sudo udevadm control --reload-rules
EOF
