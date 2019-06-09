import cv2

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

for i in range(10, 11):
    sn = "SN0" + ("0" if i < 10 else "") + str(i)
    leftcam = cv2.VideoCapture('O:/Dropbox/Videos_LeftCamera/LeftVideo%s_comp.avi' % sn)
    rightcam = cv2.VideoCapture('O:/Dropbox/Video_RightCamera/RightVideo%s_comp.avi' % sn)
    l_success, l_image = leftcam.read()
    r_success, r_image = rightcam.read()
    count = 0

    l_temp = (0, 0, 0, 0)
    r_temp = (0, 0, 0, 0)

    while l_success and r_success:
        l_grey = cv2.cvtColor(l_image, cv2.COLOR_BGR2GRAY)
        r_grey = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)

        l_faces = faceCascade.detectMultiScale(
            l_grey,
            scaleFactor=1.7,
            minNeighbors=5,
            minSize=(40, 40)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        r_faces = faceCascade.detectMultiScale(
            r_grey,
            scaleFactor=1.7,
            minNeighbors=5,
            minSize=(40, 40)
            #flags = cv2.CV_HAAR_SCALE_IMAGE
        )

        print("writing frame %d" % count)
        (x, y, w, h) = (l_temp if len(l_faces) == 0 else l_faces[0])
        (x2, y2, w2, h2) = (r_temp if len(r_faces) == 0 else r_faces[0])
        cv2.imwrite("O:/Documents/DISFA/%d/%d.jpg" % (2*i - 1, count), cv2.resize(l_image[y:y+h, x:x+w], (32, 32)))
        cv2.imwrite("O:/Documents/DISFA/%d/%d.jpg" % (2*i, count), cv2.resize(r_image[y2:y2+h2, x2:x2+w2], (32, 32)))

        l_success, l_image = leftcam.read()
        r_success, r_image = rightcam.read()
        count += 1
        l_temp = (x, y, w, h)
        r_temp = (x2, y2, w2, h2)