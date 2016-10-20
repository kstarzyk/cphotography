import numpy as np
from PIL import Image
import argparse

# SRC_PATH = '../data/'
DEST_PATH = './results/'

def nparray2png(arr, name):
  img_result = Image.fromarray(arr, 'RGB')
  img_result.save(DEST_PATH+'/'+name)

def collectPixels(img, rb_reversed=False, plot_everything=False):
  width, height = len(img[0]), len(img)
  print("COLLECTING PIXELS")
  rchan = np.zeros((height, width, 3), dtype=np.uint8)
  gchan = np.empty_like(rchan)
  bchan = np.empty_like(rchan)
  for row in range(0,height):
    for col in range(0,width):
      if (row % 2 == 1 and col % 2 == 1 and not rb_reversed) or (row % 2 == 0 and col % 2 == 0 and rb_reversed): # blue
        bchan[row][col] = [0,0,img[row][col]]
      elif (row % 2 == 0 and col % 2 == 0 and not rb_reversed) or (row % 2 == 1 and col % 2 == 1 and rb_reversed): # blue
        rchan[row][col] = [img[row][col],0,0]
      else:
        gchan[row][col] = [0,img[row][col],0] # green
  print("COLLECTING PIXELS FINISHED")
  if plot_everything:
    nparray2png(rchan, 'red_ch.png')
    print("Red channel saved to red_ch.png")
    nparray2png(gchan, 'green_ch.png')
    print("Green channel saved to green_ch.png")
    nparray2png(bchan, 'blue_ch.png')
    print("Blue channel saved to blue_ch.png")
  return rchan, gchan, bchan

def demosaicing_simple(ch, w, h, pos):
  print("DEMOSAICING (SIMPLE)")
  ch[0][0] = (ch[0][1] + ch[1][0]) / 2
  for j in range(2, w-1, 2):
    ch[0][j][pos] = (int(ch[0][j-1][pos]) + int(ch[1][j][pos]) + int(ch[0][j+1][pos]) ) / 3

  for i in range(1, h-1):
    st = 1
    if i % 2 == 0:
      ch[i][0][pos] = (int(ch[i-1][0][pos]) + int(ch[i][1][pos]) + int(ch[i+1][0][pos])) / 3
      st = 2
    for j in range(st, w-1, 2):
      ch[i][j][pos] = (int(ch[i-1][j][pos]) + int(ch[i+1][j][pos]) + int(ch[i][j-1][pos]) + int(ch[i][j+1][pos])) / 4
      if i % 2 == 1:
        ch[i][w-1][pos] = (int(ch[i][w-2][pos]) + int(ch[i-1][w-1][pos]) + int(ch[i+1][w-1][pos])) / 3
  for j in range(1, w-1, 2):
    ch[h-1][j][pos] = (int(ch[h-1][j-1][pos]) + int(ch[h-2][j][pos]) + int(ch[h-1][j+1][pos]) ) / 3

  print("DEMOSAICING FINISHED")
  return ch
def demosaicing_edge(ch, pos):
  w, h = len(ch[0]), len(ch)
  print("DEMOSAICING (EDGE-DIRECTED)")
  ch[0][0] = (ch[0][1] + ch[1][0]) / 2
  for col in range(2, w-1, 2):
    ch[0][col][pos] = (int(ch[0][col-1][pos]) + int(ch[1][col][pos]) + int(ch[0][col+1][pos]) ) / 3

  for row in range(1, h-1):
    st = 1
    if row % 2 == 0:
      ch[row][0] = (int(ch[row-1][0][pos]) + int(ch[row][1][pos]) + int(ch[row+1][0][pos])) / 3
      st = 2
    for col in range(st, w-1, 2):
      vh = abs(int(ch[row][col-1][pos]) - int(ch[row][col+1][pos]))
      vv = abs(int(ch[row-1][col][pos]) - int(ch[row+1][col][pos]))
      if vh > vv:
        ch[row][col][pos] = ( int(ch[row-1][col][pos]) +  int(ch[row+1][col][pos])  ) / 2
      elif vh < vv:
        ch[row][col][pos] = ( int(ch[row][col-1][pos]) +  int(ch[row][col+1][pos])  ) / 2
      else:
        ch[row][col][pos] = (int(ch[row-1][col][pos]) + int(ch[row+1][col][pos]) + int(ch[row][col-1][pos]) + int(ch[row][col+1][pos])) / 4
    if row % 2 == 1:
      ch[row][w-1] = (int(ch[row][w-2][pos]) + int(ch[row-1][w-1][pos]) + int(ch[row+1][w-1][pos])) / 3

  for col in range(1, w-1, 2):
    ch[h-1][col][pos] = (int(ch[h-1][col-1][pos]) + int(ch[h-2][col][pos]) + int(ch[h-1][col+1][pos]) ) / 3

  print("DEMOSAICING FINISHED")
  return ch

def demoblue(ch, w, h, pos):
  ch[0][0][pos] = ch[1][1][pos]
  for col in range(1, w-1):
    if col % 2 == 1:
      ch[0][col][pos] = ch[1][col][pos]
    else:
      ch[0][col][pos] = (int(ch[1][col-1][pos]) + int(ch[1][col+1][pos])) / 2

  ch[0][w-1][pos] = ch[1][w-1][pos]

  for row in range(1, h-1):
    if row % 2 == 0:
      ch[row][0][pos] = (int(ch[row-1][1][pos]) + int(ch[row+1][1][pos])) / 2
      for col in range(1, w-1):
        if col % 2 == 0:
          ch[row][col][pos] = (int(ch[row-1][col-1][pos]) + int(ch[row-1][col+1][pos]) + int(ch[row+1][col-1][pos]) + int(ch[row+1][col+1][pos])) / 4
        else:
          ch[row][col][pos] = (int(ch[row-1][col][pos]) + int(ch[row+1][col][pos])) / 2
      ch[row][w-1][pos] = (int(ch[row-1][w-1][pos]) + int(ch[row+1][w-1][pos])) / 2
    else:
      ch[row][0][pos] = ch[row][1][pos]
      for col in range(2, w-1, 2):
        ch[row][col][pos] = (int(ch[row][col-1][pos]) + int(ch[row][col+1][pos])) / 2

  for col in range(1, w-1):
    if col % 2 == 1:
      ch[h-1][col][pos] = ch[h-2][col][pos]
    else:
      ch[h-1][col][pos] = (int(ch[1][col-1][pos]) + int(ch[1][col+1][pos])) / 2

  return ch

def demored(ch, w, h, pos):
  for col in range(1, w-1, 2):
    ch[0][col][pos] = (int(ch[0][col-1][pos]) + int(ch[0][col+1][pos])) / 2

  ch[0][w-1][pos] = ch[0][w-2][pos]
  for row in range(1, h-1):
    if row % 2 == 1:
      for col in range(0, w-1):
        if col % 2 == 0:
          ch[row][col][pos] = (int(ch[row-1][col][pos]) + int(ch[row+1][col][pos])) / 2
        else:
          ch[row][col][pos] = (int(ch[row-1][col-1][pos]) + int(ch[row-1][col+1][pos]) + int(ch[row+1][col-1][pos]) + int(ch[row+1][col+1][pos])) / 4

      ch[row][w-1][pos] = (int(ch[row-1][w-2][pos]) + int(ch[row+1][w-2][pos])) / 2
    else:
      for col in range(1, w-1,2):
        ch[row][col][pos] = (int(ch[row][col-1][pos]) + int(ch[row][col+1][pos])) / 2
      ch[row][w-1][pos] = ch[row][w-2][pos]

  for col in range(1, w-1):
    if col % 2 == 0:
      ch[h-1][col][pos] = ch[h-2][col][pos]
    else:
      ch[h-1][col][pos] = (int(ch[h-2][col-1][pos]) + int(ch[h-2][col+1][pos])) / 2

  return ch


def compose_from_channels(red, green, blue, w, h):
  r = np.empty((h,w,3), dtype=np.uint8)
  for i in range(h):
    for j in range(w):
      r[i][j][0], r[i][j][1], r[i][j][2] = red[i][j][0], green[i][j][1], blue[i][j][2]
  return r

def gamma_correction(level, img):
  print("GAMMA CORRECTION")
  coef = 1.0 / level
  corrected = 255.0 * pow(img / 255.0, coef)
  print("GAMMA CORRECTION FINISHED")
  return corrected.astype(np.uint8, copy=False)

def demosaicing(primary, start, other, edge_directed=False, rb_reversed=False, plot_everything=False):
  width, height = len(primary[0]), len(primary)
  green = demosaicing_simple(primary, width, height, 1) if not edge_directed else demosaicing_edge(primary, 1)
  red = np.empty_like(green)
  blue = np.empty_like(green)
  if not rb_reversed:
    red = demored(start, width, height, 0)
    blue = demoblue(other, width, height, 2)
  else:
    red = demoblue(start, width, height, 0)
    blue = demored(other, width, height, 2)

  if plot_everything:
    nparray2png(red, 'dem_red_ch.png')
    print("Red channel after demosaicing saved to dem_red_ch.png")
    nparray2png(green, 'dem_green_ch.png')
    print("Green channel after demosaicing saved to dem_green_ch.png")
    nparray2png(blue, 'dem_blue_ch.png')
    print("Blue channel after demosaicing saved to dem_blue_ch.png")
  return compose_from_channels(red, green, blue, width, height)

# naive aproach, slow!
# based on http://www.equasys.de/colorconversion.html
# without footroom/headroom
def ycbcr2rgb(ycbcr):
  rgb = np.empty_like(ycbcr)
  rgb = rgb.astype(np.float32, copy=False)
  coefs = [ [1.0, 0, 1.400], [1.0, -0.343, -0.711], [1.0, 1.765, 0]]
  h = len(rgb)
  w = len(rgb[0])
  for row in range(h):
    for col in range(w):
      for i in range(3):
        rgb[row][col][i] = np.dot(coefs[i], [float(ycbcr[row][col][0]), float(ycbcr[row][col][1]-128), float(ycbcr[row][col][2]-128)])
        rgb[row][col][i] = max(0, min(255, rgb[row][col][i]))

  rgb = rgb.astype(np.uint8, copy=False)
  return rgb

# naive aproach, slow!
# based on http://www.equasys.de/colorconversion.html
# without footroom/headroom

def rgb2ycbcr(rgb):
  ycbcr = np.empty_like(rgb, )
  coefs = [ [0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
  h = len(rgb)
  w = len(rgb[0])
  for row in range(h):
    for col in range(w):
      for i in range(3):
        ycbcr[row][col][i] = np.dot(coefs[i], rgb[row][col])
      ycbcr[row][col][1] += 128
      ycbcr[row][col][2] += 128
  return ycbcr

def median_filter(img):
  print("MEDIAN FILTER")
  ycbcr = rgb2ycbcr(img)
  #img_arr = Image.fromarray(img)
  #ycbcr_img = img_arr.convert('YCbCr')
  #ycbcr = np.asarray(ycbcr_img)
  res = np.empty_like(ycbcr)
  res[:] = ycbcr
  for row in range(0,len(res)-3,1):
      for col in range(0,len(res[0])-3,3):
          d = sorted([
          ycbcr[row][col],
          ycbcr[row][col+1],
          ycbcr[row][col+2],
          ycbcr[row+1][col],
          ycbcr[row+1][col+1],
          ycbcr[row+1][col+2],
          ycbcr[row+2][col],
          ycbcr[row+2][col+1],
          ycbcr[row+2][col+2]],
          key=lambda x:x[1])
          res[row+1][col+1] = d[4]

  #img_median_result = Image.fromarray(np.asarray(Image.fromarray(res, 'YCbCr')), 'RGB')
  img_median_result = ycbcr2rgb(res)
  nparray2png(img_median_result, 'result_after_median_wihtout_gamma.png')
  print("MEDIAN FILTER FINISHED")
  return img_median_result

def pipe(path, dest, gamma=2.2, edge_directed=False, channels_reversed=False, plot_everything=False):
  #path = SRC_PATH + path
  print("PIPE START: " + path)
  img_raw = np.asarray(Image.open(path))
  width = len(img_raw[0])
  height = len(img_raw)

  red,green,blue = collectPixels(img_raw, channels_reversed, plot_everything)
  result = demosaicing(green, red, blue, edge_directed, channels_reversed, plot_everything)
  nparray2png(result, dest)
  print("Result (demosaicing) " + dest)

  result_corrected = gamma_correction(gamma, result)
  corrected_dest = dest.replace(".png", "_corrected.png")
  nparray2png(result_corrected, corrected_dest)
  print("Result (demosaicing + gamma_correction) saved to " + corrected_dest)

  filter_result = median_filter(result)
  filter_corrected_result = gamma_correction(gamma, filter_result)
  corrected_with_filter_dest = dest.replace(".png", "_corrected_with_median_filter.png")
  nparray2png(filter_corrected_result, corrected_with_filter_dest)
  print("Result (demosaicing + median filter + gamma_correction) saved to " + corrected_with_filter_dest)



# pipe('lighthouse_RAW_noisy_sigma0.01.png', 'result.png', plot_everything=True)
#pipe('raw/signs-small.png', 'result.png', edge_directed=True, channels_reversed=True, plot_everything=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Simple image processing pipe (demosaicing + median filter + gamma correction')
  parser.add_argument('src', type=str,
                          help='A required src of img to process')
  parser.add_argument('dest', type=str,
                          help='A required destination name of img')
  parser.add_argument('--gamma', type=float, nargs='?',
                          help='Gamma correction coefficent')

  parser.add_argument('--edge_directed', action='store_true',
                          help='An edge directed switch')

  parser.add_argument('--channels_reversed', action='store_true',
      help='Pass false if mosaic is in form:\nB G B G\nG R G R')

  parser.add_argument('--print_everything', action='store_true',
                          help='An edge directed switch')

  args = parser.parse_args()
  pipe(args.src, args.dest, args.gamma, args.edge_directed, args.channels_reversed, args.print_everything)
  #print(args)

