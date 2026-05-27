import cv2
import numpy as np


class DeepfakeAnalysisService:

    def compute_similarity(self, img1, img2):

        img1 = cv2.resize(img1, (128, 128))
        img2 = cv2.resize(img2, (128, 128))

        diff = np.mean(np.abs(img1.astype(float) - img2.astype(float)))

        score = 1 / (1 + diff)

        return score

    def validate(self, left_reflection, right_reflection):

        similarity = self.compute_similarity(
            left_reflection,
            right_reflection
        )

        if similarity < 0.02:
            return {
                "status": "Possible Deepfake",
                "score": similarity
            }

        return {
            "status": "Likely Real",
            "score": similarity
        }