"""
This example shows how you can use gstreamer to grab frames from a webcam,
perform inference, and then overlay the predictions on the video display.
I personally prefer using GStreamer over cv2 because:
  1. It pushes the video capture and display into their own threads which improves performance
  2. Smaller dependency footprint. OpenCV is huge and if you're on mainstream
     linux distros you most likely already have GStreamer installed

MacOS support is possible too I just don't own a mac to test with.
https://stackoverflow.com/questions/71612540/gstreamer-cannot-find-internal-camera-on-a-mac

GStreamer Installation:
  https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c

load weights from
https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth
a rough copy of
https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
"""
import logging
import os
import numpy as np
np.set_printoptions(suppress=True)
from tinygrad.tensor import Tensor
from extra.utils import fetch
from models.efficientnet import EfficientNet
from PIL import Image
from gstreasy import GstPipeline  # pip install gstreasy

# configure gstreasy logging if DEBUG=1
if int(os.getenv("DEBUG", 0)):
  fmt = "%(levelname)-6.6s | %(name)-20s | %(asctime)s.%(msecs)03d | %(threadName)s | %(message)s"
  dmt_fmt = "%d.%m %H:%M:%S"
  log_handler = logging.StreamHandler()
  log_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=dmt_fmt))
  logging.basicConfig(level=logging.DEBUG, handlers=[log_handler])
  log = logging.getLogger(__name__)

def infer(model, img):
  # preprocess image
  aspect_ratio = img.size[0] / img.size[1]
  img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))

  img = np.array(img)
  y0,x0=(np.asarray(img.shape)[:2]-224)//2
  retimg = img = img[y0:y0+224, x0:x0+224]

  # if you want to look at the image
  """
  import matplotlib.pyplot as plt
  plt.imshow(img)
  plt.show()
  """

  # low level preprocess
  img = np.moveaxis(img, [2,0,1], [0,1,2])
  img = img.astype(np.float32)[:3].reshape(1,3,224,224)
  img /= 255.0
  img -= np.array([0.485, 0.456, 0.406]).reshape((1,-1,1,1))
  img /= np.array([0.229, 0.224, 0.225]).reshape((1,-1,1,1))

  # run the net
  out = model.forward(Tensor(img)).cpu()

  # if you want to look at the outputs
  """
  import matplotlib.pyplot as plt
  plt.plot(out.data[0])
  plt.show()
  """
  return out, retimg

if __name__ == "__main__":
  # instantiate my net
  model = EfficientNet(int(os.getenv("NUM", "0")))
  model.load_from_pretrained()

  # category labels
  import ast
  lbls = fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt")
  lbls = ast.literal_eval(lbls.decode('utf-8'))

  # sane defaults for most webcams
  caps = 'video/x-raw, width=640, height=480, framerate=30/1'
  # your webcam is probably at /dev/video0 | pass WEBCAM=1 for /dev/video1
  device = int(os.getenv("WEBCAM", 0))
  cmd = f"""
    v4l2src device=/dev/video{device} ! tee name=t
    t. ! queue ! {caps} ! videoconvert ! video/x-raw,format=RGB
       ! appsink emit-signals=true sync=false
    t. ! queue ! {caps} ! videoconvert
       ! textoverlay name=overlay font-desc='Monospace 18'
       ! autovideosink sync=false
  """
  # We can produce buffers faster than we can consume them, so just drop
  # old buffers by passing leaky=True, qsize=1 args.
  with GstPipeline(cmd, leaky=True, qsize=1) as pipeline:
    text_overlay = pipeline.get_by_name("overlay")
    while pipeline:
      buffer = pipeline.pop()
      if buffer:
        img = Image.fromarray(buffer.data)
        out, retimg = infer(model, img)
        s = f"{np.argmax(out.data)} {np.max(out.data)} {lbls[np.argmax(out.data)]}"
        text_overlay.set_property("text", s)
