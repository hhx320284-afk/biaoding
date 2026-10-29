#!/usr/bin/env bash
# 用户点击界面按钮后，经系统管理员认证，仅启用所选 SocketCAN 网卡。
# 不发送机械臂命令，不写系统持久配置，不中断已经启用的接口。
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9_.:-]{1,15}$ ]]; then
    printf '%s\n' '请提供有效的 CAN 接口名。' >&2
    exit 1
fi
channel="$1"
interface="/sys/class/net/$channel"
if [[ ! -r "$interface/type" || "$(cat "$interface/type")" != 280 ]]; then
    printf '%s\n' "未找到 SocketCAN 接口 $channel，请插好适配器后刷新接口。" >&2
    exit 1
fi
flags="$(cat "$interface/flags")"
if (( flags & 1 )); then
    printf '%s\n' "$channel 已启用，未更改现有设置。"
    exit 0
fi
if [[ $EUID -ne 0 ]]; then
    printf '%s\n' '需要系统管理员认证才能启用 CAN 接口。' >&2
    exit 1
fi
/usr/sbin/ip link set dev "$channel" type can bitrate 1000000
/usr/sbin/ip link set dev "$channel" up
printf '%s\n' "$channel 已启用（1 Mbps），本次接入有效。"
