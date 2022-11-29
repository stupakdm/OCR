import cv2
import numpy as np

class Update:

    def resize_image(self, img):
        H, W = img.shape[:2]
        #print(img, img.shape)
        img = img[:H//8, :, :]
        #show_img(img, 'after resize')
        return img



    def corner_images(self, img):
        operatedImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        operatedImage = np.float32(operatedImage)
        dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

        dest = cv2.dilate(dest, None)

        #print(dest.shape)
        #print(img.shape)
        # print(dest)
        ind_y, ind_x = np.where(dest > 0.01 * dest.max())
        ind_y = np.sort(ind_y)
        ind_x = np.sort(ind_x)
        H, W = img.shape[:2]

        if ind_y[0] > 4:
            ind_y -= 4
        if ind_y[-1] < H - 4:
            ind_y[-1] += 4

        if ind_x[0] > 4:
            ind_x -= 4
        if ind_x[-1] < W - 4:
            ind_x[-1] += 4

        if (((ind_x[0] - 0)+(W-ind_x[-1])) > abs(ind_x[-1]-ind_x[0])//2) or (((ind_y[0] - 0)+(H-ind_y[-1])) > abs(ind_y[-1]-ind_y[0])//2):
            print('Cropp width and height: ', H, W)
            img = img[ind_y[0]:ind_y[-1], ind_x[0]:ind_x[-1]]
            img = self.gaussian_blur(img, kernel=(3, 3))
            # print(np.where(dest > 0.01* dest.max()))
            # img[dest > 0.01 * dest.max()] = [0,0,255]
        return img


    def getStructure(self, k=(5, 5), choice='RECT'):
        return cv2.getStructuringElement(cv2.__dict__['MORPH_' + choice], k)

    def morphology(self, image, k=(5, 5), choice='OPEN'):
        # kernel = np.ones(k, np.uint8)
        kernel = k
        opening = cv2.morphologyEx(image, cv2.__dict__['MORPH_' + choice], kernel)
        return opening

    def black_white(self, image):
        # new_image = cv2.threshold(image, )
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return img

    def gaussian_blur(self, image, kernel=(5, 5), sigma=0.0):
        gaus = cv2.GaussianBlur(image, kernel, sigma)
        return gaus


    def unsharp_mask(self, image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        # widthKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
        try:
            blurred = self.gaussian_blur(image, kernel_size, sigma)

            # blurred = self.median_blur(image)
            # blurred = self.averaging_blur(image)
            sharpened = float(amount + 1) * image - float(amount) * blurred
            sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
            sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
            sharpened = sharpened.round().astype(np.uint8)
            if threshold > 0:
                low_contrast_mask = np.absolute(image - blurred) < threshold
                np.copyto(sharpened, image, where=low_contrast_mask)
            return sharpened
        except:
            return image


def show_img(img, title):
# print(img.shape)
    # print(img)
    #plt.imshow(img)
    #plt.title(title)
    #plt.show()
    pass