import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/arlo/Ros_projects/Ros2/ass6/src/install/asse6'
