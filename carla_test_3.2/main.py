import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import time
import subprocess
from pathlib import Path


def main():
    subprocess.Popen(['gnome-terminal', '-x', 'sh', '/home/kola/carla/CarlaUE4.sh'])
    time.sleep(10.0)

if __name__ == '__main__':
    main()

