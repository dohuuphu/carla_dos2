import cv2
import numpy as np
import matplotlib.pyplot as plt
from LightGlue.lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from LightGlue.lightglue.utils import image_to_torch, rbd
from LightGlue.lightglue import viz2d

class GM(object):
  def __init__(self):
      self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
      self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
      self.COUNT = 0

  def detect_and_match_features_DL(self,img1, img2):
      
      image0 = image_to_torch(img1).cuda()
      image1 = image_to_torch(img2).cuda()
      #extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
      feats0 = self.extractor.extract(image0)  # auto-resize the image, disable with resize=None
      feats1 = self.extractor.extract(image1)
      matches01 = self.matcher({'image0': feats0, 'image1': feats1})
      feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
      matches = matches01['matches']  # indices with shape (K,2)
      points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
      points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
      return  points0,points1

  def estimate_translation_dl(self,kp1,xy1, kp2,xy2):
      if kp1.shape[0] < 2:
          return None, None
      else:
        src_pts = np.float32(self.translate_keypoints_dl(kp1,xy1[0],xy1[1])).reshape(-1, 2)
        dst_pts = np.float32(self.translate_keypoints_dl(kp2,xy2[0],xy2[1])).reshape(-1, 2)

        translations = dst_pts - src_pts

        # Use RANSAC to filter out outliers
        translation_vector, inliers = self.ransac_translation(translations)

        return translation_vector, inliers

  def translate_keypoints_dl(self,kp1, dx, dy):
      translated_kp = []
      for i in range(kp1.shape[0]):
          x, y = kp1[i]
          x=x.cpu().numpy()
          y=y.cpu().numpy()
          translated_kp.append((x + dx, y + dy))
      return np.array(translated_kp)

  def ransac_translation(self,translations, threshold=0.5, max_iterations=100):
      best_translation = None
      best_inliers_count = 0
      best_inliers = None

      for _ in range(max_iterations):
          idx = np.random.choice(len(translations), 1, replace=False)
          translation_candidate = translations[idx]

          inliers = np.linalg.norm(translations - translation_candidate, axis=1) < threshold
          inliers_count = np.sum(inliers)

          if inliers_count > best_inliers_count:
              best_translation = translation_candidate
              best_inliers_count = inliers_count
              best_inliers = inliers

      return best_translation, best_inliers_count


  def detect_and_match_features_sift(self,img1, img2):
      
      sift = cv2.SIFT_create()
      kp1, des1 = sift.detectAndCompute(img1, None)
      kp2, des2 = sift.detectAndCompute(img2, None)    
      bf = cv2.BFMatcher()
      good_matches = []

      if des1 is not None and des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)
        try:
          for m, n in matches:
              if m.distance < 0.95 * n.distance:
                  good_matches.append(m)
        except:
          pass
      
      return kp1, kp2, good_matches

  def find_translate(self,img,img1, box,box1):
      center_y, center_x, width, height, angle = box[:5]

      center_y1, center_x1, width1, height1, angle1 = box1[:5]

      # Tạo một mask ảnh có cùng kích thước với ảnh gốc và vẽ hình chữ nhật lên đó
      mask = np.zeros_like(img, dtype=np.uint8) 
      rect = ((center_x, center_y), (width*2+15, height*2+15), -90-np.degrees(angle))
      box_points = cv2.boxPoints(rect).astype(int)
      cv2.fillPoly(mask, [box_points], 255)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        img_crop = img[y:y+h,x:x+w]
        

      mask1 = np.zeros_like(img1, dtype=np.uint8) 
      rect1 = ((center_x1, center_y1), (width1*2+15, height1*2+15), -90-np.degrees(angle1))
      box_points1 = cv2.boxPoints(rect1).astype(int)
      cv2.fillPoly(mask1, [box_points1], 255)
      mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)
      contours, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      for cnt in contours:
        x1,y1,w1,h1 = cv2.boundingRect(cnt)
        img_crop1 = img1[y1:y1+h1,x1:x1+w1]
      kp1 = None
      
      #resize
      h,w,_ = img_crop.shape
      h1,w1,_ = img_crop1.shape
      img_crop = cv2.resize(img_crop, (int(w*1.25), int(h*1.25)))

      img_crop1 = cv2.resize(img_crop1,  (int(w*1.25), int(h*1.25)))

      kp1, kp2, matches = self.detect_and_match_features_sift(img_crop, img_crop1)
      #kp1, kp2 = self.detect_and_match_features_DL(img_crop, img_crop1)
  
      if kp1 is not None:
        #axes = viz2d.plot_images([img_crop, img_crop1])
        #viz2d.plot_matches(kp1, kp2, color="lime", lw=0.2,axes=axes)
        #img3 = cv2.drawMatches(img_crop,kp1,img_crop1,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.COUNT +=1 
        plt.savefig('./visualize/'+str(self.COUNT)+'OriginalFrame.jpg')
        T, inliers = self.estimate_translation(kp1,(x,y), kp2,(x1,y1), matches)
        #T, inliers = self.estimate_translation_dl(kp1,(x,y), kp2,(x1,y1))
        #if  T is not None:
        #    cv2.putText(img3, str(np.round(inliers, 1)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,0,0), 1, cv2.LINE_AA)
        if T is not None:
          T = T if inliers > 10 else None
        
          #cv2.imwrite('./visualize/'+str(self.COUNT)+'OriginalFrame.jpg', img3)
      return T

  def detect_and_match_features(self,img1, img2):
      orb = cv2.ORB_create()
      kp1, des1 = orb.detectAndCompute(img1, None)
      kp2, des2 = orb.detectAndCompute(img2, None)
      
      bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
      matches = bf.match(des1, des2)
      
      matches = sorted(matches, key=lambda x: x.distance)
      
      return kp1, kp2, matches

  def translate_keypoints(self,kp1, dx, dy):
      translated_kp = []
      for kp in kp1:
          x, y = kp
          translated_kp.append((x/1.25 + dx, y/1.25 + dy))
      return np.array(translated_kp)

  def estimate_translation(self,kp1,xy1, kp2,xy2, matches):
      if len(matches) < 2:
          return None, None
      else:
        src_pts = np.float32(self.translate_keypoints([kp1[m.queryIdx].pt for m in matches],xy1[0],xy1[1])).reshape(-1, 2)
        dst_pts = np.float32(self.translate_keypoints([kp2[m.trainIdx].pt for m in matches],xy2[0],xy2[1])).reshape(-1, 2)

        translations = dst_pts - src_pts

        # Use RANSAC to filter out outliers
        translation_vector, inliers = self.ransac_translation(translations)

        return translation_vector, inliers
      


  # img1 = cv2.imread('/mnt/HDD1/phudh/course/calar/IDS_s24/HW0/results/DOS/TFPP/1orignal/0096.png', cv2.IMREAD_GRAYSCALE)
  # img2 = cv2.imread('/mnt/HDD1/phudh/course/calar/IDS_s24/HW0/results/DOS/TFPP/1orignal/0097.png', cv2.IMREAD_GRAYSCALE)

  # kp1, kp2, matches = detect_and_match_features(img1, img2)
  # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  # H, mask = estimate_motion(kp1, kp2, matches)
  # print(H)
  # img2_estimated = apply_motion_estimation(img1, H)

  # # Hiển thị kết quả
  # #cv2.imshow('Original Frame', img2)
  # cv2.imwrite('OriginalFrame.jpg', img2)
  # #cv2.imshow('Original Frame 22', img3)
  # plt.imshow(img3)
  # plt.show()
  # #cv2.imshow('Estimated Frame', img2_estimated)
  # cv2.imwrite('EstimatedFrame.jpg', img2_estimated)

  # #cv2.waitKey(0)
  # #cv2.destroyAllWindows()