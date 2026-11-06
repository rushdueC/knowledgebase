# -*- coding: utf-8 -*-
"""
打印本机可访问地址（给 启动控制台.bat 用）

为什么不写在 .bat 里：
    cmd 的 for/f + 引号嵌套极其容易出错（实测把 LAN IP 取丢了）。
    交给 Python 一行搞定，跨机器也稳。

用法：
    python show_urls.py [port]
"""
import socket
import sys


def lan_ip() -> str:
    """拿本机在局域网里的 IP；拿不到就返回空串"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不会真的发包，只是让系统选一条出口路由
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:                            # noqa: BLE001
        return ""


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "8501"
    print(f"  Local URL    : http://localhost:{port}")
    ip = lan_ip()
    if ip and not ip.startswith("127."):
        print(f"  Network URL  : http://{ip}:{port}   (同一局域网的其他设备可用)")
    else:
        print("  Network URL  : （没检测到局域网 IP，只能本机访问）")


if __name__ == "__main__":
    main()
