#!/usr/bin/env python
#vim:set et sw=4 ts=4:
import time
import os
from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Pool
import numpy as np

def parse_size(S:str) -> int:
    s=S.lower()
    try:
        if s.endswith('k'): b= int(s[:-1]) * 1024 ;
        elif s.endswith('m'): b= int(s[:-1]) * 1024 **2 ;
        elif s.endswith('g'): b= int(s[:-1]) * 1024 **3 ;
        else: b=int(s) ;
    except ValueError:
        raise ArgumentTypeError(f'invalid size: "{S}", should be number or number with suffix K, M, or G')
    return b

def entropy2(labels, base=None):
  from math import e, log
  #https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
  """ Computes entropy of label distribution. """
  n_labels = len(labels)
  if n_labels <= 1:
    return 0
  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)
  if n_classes <= 1:
    return 0
  ent = 0.
  # Compute entropy
  base = e if base is None else base
  for i in probs:
    ent -= i * log(i, base)
  return ent

def read_blocks(f, block_size):
    while True:
        b = f.read(block_size)
        if len(b) < block_size: # ==0 to calculate last block too
            break
        yield b

def read_file(filename, block_size):
    f=open(filename, 'rb')
    return read_blocks(f, block_size)

def read_mmap(filename, block_size):
    import mmap
    f=open(filename, 'rb')
    m=mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    return read_blocks(m, block_size)

def get_entropy(data):
    return entropy2(np.frombuffer(data, dtype='uint8'), base=2)

def tqdm(iterable, total): # barebones tqdm clone
    nextpt=time.time()
    i=0
    for it in iterable:
        i += 1
        now=time.time()
        if now > nextpt:
            nextpt += 0.25
            print(f'\033[1K\r{i}/{total} {i/total*100:02.2f}%', end='')
        yield it
    print(f'\033[1K\r{i}/{total} {i/total*100:02.2f}%')

def filesize(path): # works with both regular and block devices
    with open(path, 'rb') as f:
        f.seek(0, 2)
        return f.tell()

def d2xy(n, d):
    """ get (x,y) coordinate on hilbert curve. copy paste translate from wikipedia """
    t=d
    x,y=0,0
    s=1
    while s < n:
        rx = 1 & (t >> 1)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s-1 - x
                y = s-1 - y
            x,y = y,x
        x += s * rx
        y += s * ry
        t >>= 2
        s *= 2
    return x,y

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def main():
    ap=ArgumentParser()
    ap.add_argument('-b', '--block-size', type=parse_size, default='1M')
    ap.add_argument('-p', '--processes', type=int, default=os.cpu_count())
    ap.add_argument('-g', '--graph', default=False, action='store_true')
    ap.add_argument('-m', '--mmap', default=False, action='store_true')
    ap.add_argument('-G', '--hilbert-curve-heatmap', default=False, action='store_true')
    ap.add_argument('-c', '--output-csv', type=str, metavar='FILENAME')
    ap.add_argument('filename')
    args=ap.parse_args()
    print('block size:', args.block_size)

    t0 = time.time()
    size = filesize(args.filename)
    n = size // args.block_size
    
    if n < 1:
        print("block size > file size. will not")
        exit(42)

    if args.mmap:
        blocks = read_mmap(args.filename, args.block_size)
    else:
        blocks = read_file(args.filename, args.block_size)

    print('processes:', args.processes)
    with Pool(args.processes) as p:
        entropys = []
        for x in tqdm(p.imap(get_entropy, blocks, chunksize=128), total=n):
            entropys += [x]

    t1 = time.time()
    dt = t1 - t0
    n = len(entropys)
    print(n, 'blocks')
    print(f"{dt:.2f} s")
    print(f"{args.block_size * n / (1024**2) / dt:.2f} MiB/s")
    print(f'entropy min: {min(entropys)} max: {max(entropys)}')

    if args.output_csv:
        with open(args.output_csv, 'w') as f:
            print('blocknr,entropyBase2', file=f)
            for i,e in enumerate(entropys):
                print(f"{i},{e}", file=f)

    if args.graph:
        import matplotlib.pyplot as plt
        plt.plot(entropys)
        plt.ylabel('entropyBase2')
        plt.xlabel('blocknr')
        plt.show()

    if args.hilbert_curve_heatmap:
        s=next_power_of_2(int(np.sqrt(n)))
        raster=np.zeros((s,s))
        entropys_interp = np.interp(np.arange(0,s*s)/(s*s-1), np.arange(0,n)/(n-1), entropys)
        #lo,hi = min(entropys) , max(entropys)
        lo,hi = 0,8
        for i in range(raster.size):
            x,y=d2xy(s, i)
            raster[y,x] = (entropys_interp[i] - lo) / (hi - lo)
        import matplotlib.pyplot as plt
        plt.imshow(raster)
        plt.show()

if __name__=="__main__":
    main()

