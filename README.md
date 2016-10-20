## Basic image processing (demosaicing + medium filter + gamma correction)
*DISCLAIMER*: It's student project. DON'T treat it as reference of good code or how to 'make things done' in any way, please!

You can use both python and python3

## Dependencies
- PIL
- numpy

## Usage
```
usage: pipe.py [-h] [--gamma [GAMMA]] [--edge_directed] [--channels_reversed]
               [--print_everything]
               src dest

Simple image processing pipe (demosaicing + median filter + gamma correction

positional arguments:
  src                  A required src of img to process
  dest                 A required destination name of img

optional arguments:
  -h, --help           show this help message and exit
  --gamma [GAMMA]      Gamma correction coefficent
  --edge_directed      An edge directed switch
  --channels_reversed  Pass false if mosaic is in form: B G B G G R G R
  --print_everything   An edge directed switch
```

## Examples
(python3 is slight faster)
```
$ python3 pipe.py lighthouse_RAW.png lighthouse_result.png --gamma 2.2 --print_everything --edge_directed
$ python3 pipe.py signs-small.png signs_result.png --gamma 1.9 --print_everything --edge_directed --channels_reversed

```
