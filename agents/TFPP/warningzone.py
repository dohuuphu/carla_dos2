import numpy as np
import matplotlib.pyplot as plt
import torch

def select_largest_bbox_rightImg(det, width):
    bboxes = []
    for *xyxy, conf, cls in reversed(det):
      c = int(cls)  # integer class
      if c in [2,5,7]: # car, bus, truck
        bboxes.append(xyxy)
          
    
    middle_x = width // 2

    # find box on the right
    filtered_bboxes = [
        bbox for bbox in bboxes
        if (bbox[2] >= middle_x )
    ]

    if not filtered_bboxes:
        return None  

    largest_bbox = max(filtered_bboxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3]-bbox[1]))
    return largest_bbox

def average_depth_in_bbox(depth_map, bbox):

    x_min, y_min, x_max, y_max = map(int, bbox)
    cropped_depth_map = depth_map[0][y_min:y_max+1, x_min:x_max+1]
    average_depth = torch.mean(cropped_depth_map)

    return average_depth.item()

def get_right_boxes_and_Mindistances(target_box, ref_boxes):

    target_x_min = target_box[0]
    target_center_y = (target_box[1] + target_box[3]) / 2

    right_boxes_and_distances = []
    min_distance = float('inf')

    for ind, ref_box in enumerate(ref_boxes):
        ref_x_min = ref_box[0]
        ref_center_y = (ref_box[1] + ref_box[3]) / 2
        
        if ref_x_min > target_x_min:
            x_distance = ref_x_min - target_x_min
            y_distance = abs(ref_center_y - target_center_y)
            distance = np.sqrt(x_distance ** 2 + y_distance ** 2)
            distance = x_distance
            if distance < min_distance:
                min_distance = distance
                closest_box_id = ind

    return min_distance

def get_left_boxes_and_Maxdistances(target_box, ref_boxes):

    target_x_min = target_box[0]
    target_center_y = (target_box[1] + target_box[3]) / 2

    max_distance = 0

    for ind, ref_box in enumerate(ref_boxes):
        ref_x_min = ref_box[0]
        ref_center_y = (ref_box[1] + ref_box[3]) / 2
        
        if ref_x_min < target_x_min:
            x_distance =  target_x_min - ref_x_min
            y_distance = abs(ref_center_y - target_center_y)
            distance = np.sqrt(x_distance ** 2 + y_distance ** 2)
            distance = x_distance
            if distance > max_distance:
                max_distance = distance
                closest_box_id = ind

    return  max_distance

def get_person_info(det):
    info = {'person':[]}
    for *xyxy, conf, cls in reversed(det):
        c = int(cls)
        if c == 0:  
            info['person'].append(xyxy)
    return info

def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

def find_intersection(pred_semantic, y):

    # find line segment
    # 'red:sky', 'green:car', 'blue:road', 'yellow', 'purple:sidewalk', 'cyan:line'
    print('a ', len(pred_semantic))
    pre_semantic_image = pred_semantic
    c,h,w = pre_semantic_image.shape
    class_map = np.argmax(pre_semantic_image, axis=-3) #shape 256, 1024 
    line_segment = np.where(class_map == 5, 1, 0)
    binary_image = np.uint8(line_segment) * 255

    # Save the binary image using matplotlib
    plt.imsave("line_segment_image.png", binary_image, cmap='gray')

    (x1, y1), (x2, y2) = line_segment
    if y1 == y2:  # Đoạn thẳng nằm ngang
        return None
    
    t = (y - y1) / (y2 - y1)
    if 0 <= t <= 1:
        x_intersect = x1 + t * (x2 - x1)
        return x_intersect, y
    return None

def create_2d_cover(bbox, line_segment):
    """
    Tạo vùng 2D cover từ bbox và đoạn thẳng.
    
    bbox: tuple (x_min, y_min, x_max, y_max)
    line_segment: tuple ((x1, y1), (x2, y2))
    return: tuple (x_min, y_min, x_max, y_max) của vùng cover
    """
    center_x, center_y = calculate_center(bbox)
    intersection = find_intersection(line_segment, center_y)
    
    if intersection:
        x_intersect, y_intersect = intersection
        cover_bbox = (min(center_x, x_intersect), bbox[1], max(center_x, x_intersect), bbox[3])
    else:
        cover_bbox = bbox  # Nếu không có giao điểm, vùng cover chính là bbox ban đầu
    
    return cover_bbox



