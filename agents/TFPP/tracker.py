
import os
from copy import deepcopy
from filterpy.kalman import KalmanFilter
import cv2
from collections import deque
from shapely.geometry import Polygon
import numpy as np
import math

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
import transfuser_utils as t_u


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def get_corner(box):
	"""
	Get the coordinate of 
 	[top_left, top_right, bottom_right, bottom_left] from 
  x, y, extent_x, extenx_y, and yaw.
  
  return shape: [4, 2]
 	"""
	x, y, extent_x, extent_y, yaw = box
	half_e_x = extent_x
	half_e_y = extent_y
    
    # Compute corner coordinates relative to the center
	corners_rel = np.array([[-half_e_x, -half_e_y],  # bottom_left
                            [half_e_x, -half_e_y],   # bottom_right
                            [half_e_x, half_e_y],    # top_right
                            [-half_e_x, half_e_y]])  # top_left
    
    # Rotation matrix
	R = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]])
    
    # Rotate corner coordinates
	corners_rot = np.dot(R, corners_rel.T).T
    
    # Translate to the given center (x, y)
	corners = corners_rot + np.array([x, y])
    
	return corners

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

def iou_bbs(bb1, bb2):
	poly1 = Polygon(get_corner(bb1))
	poly2 = Polygon(get_corner(bb2))
	iou = poly1.intersection(poly2).area / poly2.union(poly1).area
	return iou

def iou_batch_rotated(bb_test, bb_gt):
    num_test = bb_test.shape[0]
    num_gt = bb_gt.shape[0]
    iou_matrix = np.zeros((num_test, num_gt))
    for i in range(num_test):
        for j in range(num_gt):
            iou = iou_bbs(bb_test[i], bb_gt[j])
            iou_matrix[i, j] = iou

    return iou_matrix

def convert_x_to_rotated_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x, y, s, r, yaw] and 
    returns it in the form [x, y, x_extent, y_extent, yaw] where x, y is the center, 
    x_extent is the width, y_extent is the height, and yaw is the rotation angle.
    """
    cx, cy, s, r, yaw = x[:5]
    
    # Calculate width and height from scale and aspect ratio
    w = np.sqrt(s * r)
    h = s / w
    
    if score is None:
        return np.array([cx, cy, w, h, yaw]).reshape((1, 5))
    else:
        return np.array([cx, cy, w, h, yaw, score]).reshape((1, 6))

def convert_rotated_bbox_to_z(bbox):
    """
    Takes a rotated bounding box in the form [x, y, x_extent, y_extent, yaw]
    and returns z in the form [x, y, s, r] where x, y is the center of the box,
    s is the scale (area), and r is the aspect ratio.
    
    Args:
    bbox (list or np.ndarray): Rotated bounding box in the format [x, y, x_extent, y_extent, yaw]
    
    Returns:
    np.ndarray: Converted bounding box in the format [x, y, s, r]
    """
    x, y, x_extent, y_extent, yaw = bbox[:5]
    s = x_extent * y_extent
    r = x_extent / float(y_extent)    
    return np.array([x, y, s, r,yaw]).reshape((5, 1))

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=5) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:5] = convert_rotated_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.history_det = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    if len(self.history) >5:
      self.history = self.history[-4:]
    self.history_det.append(bbox)
    if len(self.history_det) >5:
      self.history_det = self.history_det[-4:]
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_rotated_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_rotated_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_rotated_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch_rotated(detections[:,:5], trackers[:,:5])

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=3, min_hits=0, iou_threshold=0.1):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4]]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          if len(trk.history_det) > 2:
            ret.append(np.concatenate((d,[trk.id+1],trk.history_det[-2][:5],trk.history_det[-3][:5])).reshape(1,-1))
          else:
            ret.append(np.concatenate((d,[trk.id+1],[-10],[-10],[-10],[-10],[-10],[-10],[-10],[-10],[-10],[-10])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))


