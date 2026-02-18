import sys

if sys.platform == 'win32':
    from tracking.devices_windows import VirtualDeviceManager
else:
    from tracking.devices_linux import VirtualDeviceManager

__all__ = ['VirtualDeviceManager']
