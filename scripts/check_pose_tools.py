import sys
try:
    import torchvision
    print("torchvision:", torchvision.__version__)
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    print("keypointrcnn_resnet50_fpn is available!")
except Exception as e:
    print("torchvision error:", e)

try:
    from mediapipe.tasks.python import vision
    print("mediapipe vision:", dir(vision))
except Exception as e:
    print("mediapipe error:", e)
