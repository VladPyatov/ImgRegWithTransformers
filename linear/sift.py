import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EuclideanTransform


class SIFT:
    def __init__(self, matcher='BruteForce', root_sift=True, min_matches=3, **kwargs):
        """SIFT image registration

        :param matcher:
        :param root_sift:
        :param min_matches:
        :param kwargs: sift parameters - see help(cv2.xfeatures2d.SIFT_create) for details
        """
        # initialize the feature detector
        self.Detector = cv2.SIFT_create(**kwargs)

        # initialize the keypoint matcher
        self.Matcher = cv2.DescriptorMatcher_create(matcher)

        # initialize rootSift state
        self.RootSift = root_sift

        # initialize minimum number of keypoint matches
        self.MinMatches = min_matches

    def extract_keypoints(self, image):
        """Extract keypoints

        :param ndarray image: image
        :return: (kpsA, featuresA), (kpsB, featuresB)
        """
        # extract features from each of the keypoint regions in the images
        (kps, features) = self.Detector.detectAndCompute(image, None)

        # use Hellinger distance instead of Euclidian
        if self.RootSift:
            features /= (features.sum(axis=1, keepdims=True) + 1e-7)
            features = np.sqrt(features)

        return kps, features

    def match_keypoints(self, kpsA, featuresA, kpsB, featuresB, verbose=0):
        """Match keypoints

        :param kpsA: keypoints of the first image
        :param featuresA: features of the first image
        :param kpsB: keypoints of the second image
        :param featuresB: features of the second image
        :return: ptsA, ptsB (corresponding points)
        """
        # match the keypoints using the Euclidean/Hellinger distance and initialize the list of actual matches
        rawMatches = self.Matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        if rawMatches is not None:
            # loop over the raw matches
            for m in rawMatches:
                # ensure the distance passes David Lowe's ratio test
                if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                    matches.append((m[0].queryIdx, m[0].trainIdx))

            # show some diagnostic information
            if verbose:
                print("[INFO] # of matched keypoints: {}".format(len(matches)))

            # construct and return the two sets of points
            ptsA = np.float32([kpsA[i].pt for (i, _) in matches])
            ptsB = np.float32([kpsB[j].pt for (_, j) in matches])
            return ptsA, ptsB
        else:
            print("[ERROR] not enough matches to process")
            return None, None

    def extract_and_match_keypoints(self, imageA, imageB, verbose=0):
        """Extract and match keypoints

        :param ndarray imageA: target image
        :param ndarray imageB: source image
        :return: ptsA,ptsB (corresponding points)
        """
        kpsA, featuresA = self.extract_keypoints(imageA)
        kpsB, featuresB = self.extract_keypoints(imageB)

        if verbose:
            print("[INFO] # of keypoints from first image: {}".format(len(kpsA)))
            print("[INFO] # of keypoints from second image: {}".format(len(kpsB)))

        return self.match_keypoints(kpsA, featuresA, kpsB, featuresB, verbose=verbose)

    def registration(self, imageA, imageB, scale_factor=None, scale_threshold=3000, max_iters=10000,
                     type="R", border=(0, 0, 0), verbose=0, pts=None, return_only_matrix=True):
        """ Linear image registration

        :param ndarray imageA: target image
        :param ndarray imageB: source image
        :param float scale_factor: scaling factor - if None calculates automatically, if -1 preserve original size
        :param float scale_threshold: scaling factor stop threshold
        :param int max_iters: number of maximum iterations for RANSAC
        :param str type: transformation type (R - rigid, S - similarity, A - affine)
        :param (int, int, int) border: color to fill the border
        :param int verbose: whether to show additional information
        :return: registered image and transformation matrix
        """
        # preserve the original images
        origA = imageA.copy()
        origB = imageB.copy()

        # calculate scale_factor automatically if needed
        if scale_factor is None:
            min_dimension = min(imageA.shape[:2])
            if min_dimension > scale_threshold:
                scale_factor = 1
                while min_dimension > scale_threshold:
                    scale_factor *= 2
                    min_dimension /= 2
                if verbose:
                    print(f"[INFO] automatic scale factor = {scale_factor}")

        # resize images if scale factor > 1
        if scale_factor is not None and scale_factor > 1:
            imageA = cv2.resize(imageA,
                                dsize=(round(imageA.shape[1] / scale_factor), round(imageA.shape[0] / scale_factor)),
                                interpolation=cv2.INTER_CUBIC)
            imageB = cv2.resize(imageB,
                                dsize=(round(imageB.shape[1] / scale_factor), round(imageB.shape[0] / scale_factor)),
                                interpolation=cv2.INTER_CUBIC)

        # convert images to grayscale if they're not
        if len(imageA.shape) != 2:
            imageA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
        if len(imageB.shape) != 2:
            imageB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

        # get corresponding keypoints
        if pts != None:
            ptsA, ptsB = pts
            if scale_factor is not None and scale_factor > 1:
                ptsA /= scale_factor
                ptsB /= scale_factor
        else:
            ptsA, ptsB = self.extract_and_match_keypoints(imageA, imageB, verbose=verbose)

        # check to see if there are enough matches to process
        if len(ptsA) >= self.MinMatches:
            # compute the transformation between the two sets of points
            if verbose:
                print(f"[INFO] estimating transformation matrix of type: {type} ...")

            if type == "R":
                (model, inliers) = ransac((ptsB, ptsA), EuclideanTransform, min_samples=3, max_trials=max_iters,
                                          residual_threshold=max(imageA.shape[:2]) * 0.01)
                # get transformation matrix
                matrix = model.params[:2, :]

            elif type == "S":
                (matrix, inliers) = cv2.estimateAffinePartial2D(ptsB, ptsA, method=cv2.RANSAC, maxIters=max_iters)
            else:
                (matrix, inliers) = cv2.estimateAffine2D(ptsB, ptsA, method=cv2.RANSAC, maxIters=max_iters)

            # adjust translation parameters if scale factor > 1
            if scale_factor is not None and scale_factor > 1:
                matrix[:, -1] *= scale_factor

            if verbose:
                print("[INFO] registering images ...")
            
            if return_only_matrix:
                return matrix
            else:
                registered = cv2.warpAffine(origB, matrix, dsize=(origA.shape[1], origA.shape[0]),
                                            flags=cv2.INTER_CUBIC, borderValue=border)

                return registered, matrix
        else:
            print('[ERROR] not enough matches to process')
            if return_only_matrix:
                return None
            else:
                return None, None
