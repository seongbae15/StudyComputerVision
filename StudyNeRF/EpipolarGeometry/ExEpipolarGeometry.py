import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


def drawlines(img1, img2, lines, pts1, pts2):
    img1_c = copy.deepcopy(img1)
    img2_c = copy.deepcopy(img2)
    r, c = img1_c.shape
    img1_c = cv2.cvtColor(img1_c, cv2.COLOR_BGR2RGB)
    img2_c = cv2.cvtColor(img2_c, cv2.COLOR_BGR2RGB)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0]*c) / r[1]] )
        img1_ = cv2.line(img1_c, (x0, y0), (x1, y1), color, 1)
        img1_ = cv2.circle(img1_c, tuple(pt1), 5, color, -1)
        img2_ = cv2.circle(img2_c, tuple(pt1), 5, color, -1)
    return img1_, img2_


img_left = cv2.imread("../data/wraclaw_left.jpg", 0)
img_right = cv2.imread("../data/wraclaw_right.jpg", 0)



sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img_left, None)
kp2, des2 = sift.detectAndCompute(img_right.copy(), None)

# FLANN paramters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Find Epilines corresponding to points in right image (second image) and
#  drawing its lines on left image

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img_left, img_right, lines1, pts1, pts2)


lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img_right, img_left, lines2, pts2, pts1)

plt.subplot(121)
plt.imshow(img5)
plt.subplot(122)
plt.imshow(img3)
plt.show()
