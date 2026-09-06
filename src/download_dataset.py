import roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="xUzYSZPVpuIFmOpmzTB2")
project = rf.workspace("computer-vision-filqz").project("hard-hat-detection-62rrp")
version = project.version(1)
dataset = version.download("yolov8")
                