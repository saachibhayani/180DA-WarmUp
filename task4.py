# used the following tutorial for kmeans https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
"""
def plot_colors(hist, cent):
    start = 0
    end = 0
    myRect = np.zeros((50, 300, 3), dtype="uint8")
    tmp = hist[0]
    tmpC = cent[0]
    for (percent, color) in zip(hist, cent):
        if(percent > tmp):
            tmp = percent
            tmpC = color
    end = start + (tmp * 300) # try to fit my rectangle 50*300 shape
    cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                  tmpC.astype("uint8").tolist(), -1)
    start = end
    #rest will be black. Convert to black
    for (percent,color) in zip(hist, cent):
        end = start + (percent * 300)  # try to fit my rectangle 50*300 shape
        if(percent != tmp):
            color = [0, 0, 0]
            cv2.rectangle(myRect, (int(start), 0), (int(end), 50),
                      color, -1) #draw in a rectangle
            start = end

    return myRect

"""

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

# this code handles for a video instead
cap = cv2.VideoCapture("C:\\Users\\saach\\OneDrive\\Pictures\\Camera Roll\\WIN_20251001_18_17_50_Pro.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # optional: resize to speed up clustering
    frame = cv2.resize(frame, (200, 150))
    
    # convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))

    # apply KMeans
    clt = KMeans(n_clusters=3, n_init=10)
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    # show results
    cv2.imshow("Video", frame)
    cv2.imshow("Dominant Colors", cv2.cvtColor(bar, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



