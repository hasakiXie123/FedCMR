# -*- coding: utf-8 -*-
import numpy as np
import torch
import scipy
import scipy.spatial

def torch_cos_new(a, b):
  norm1 = torch.norm(a, dim=-1).resize(a.shape[0], 1)
  norm2 = torch.norm(b, dim=-1).resize(1, b.shape[0])
  end_norm = torch.mm(norm1, norm2)
  del norm1
  del norm2
  cos = torch.mm(a, b.T) / end_norm
  return cos

def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  """算mAP"""

  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')#Compute distance between each pair of the two collections of inputs.
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()#rank
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0#
    r = 0.0
    for j in range(k):
      if label[i] == label[order[j]]:
        r += 1
        p += (r / (j + 1))
    if r > 0:
      res += [p / r]
    else:
      res += [0]
  return np.mean(res)

def fx_calc_map_multilabel_k(image, text, labels, k=0, metric='cosine'):
    # dist = scipy.spatial.distance.cdist(image, text, metric)
    image_tensor = torch.tensor(image, dtype=torch.float32)
    text_tensor = torch.tensor(text, dtype=torch.float32)
    if torch.cuda.is_available():
      image_tensor.cuda()
      text_tensor.cuda()
    dist = torch_cos_new(image_tensor, text_tensor)
    dist = 1. - dist.numpy()
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
      if i % 10000 == 0:
        print(i)
      order = ord[i].reshape(-1)[0: k]
      tmp_label = (np.dot(labels[order], labels[i]) > 0)
      if tmp_label.sum() > 0:
        prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
        total_pos = float(tmp_label.sum())
        if total_pos > 0:
          res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)