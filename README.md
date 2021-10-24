# Car-license-plate-detection-
برنامه پلاک خوان - مخصوص پلاک ایرانی 



![22](https://user-images.githubusercontent.com/76064876/138587428-2101887a-5d98-43d6-900d-3647c2f96a67.jpg)





# plate detection
The first step in resolving plaque detection problems is to identify the plaque. In the first phase of LPR, a limiting box is produced that fits exactly where the license plate is located. This is the main condition for successful LPR. Therefore, LP diagnosis should be relatively accurate. In this article, we have used the opencv library as well as contour and edge finder to detect this bug.

![3](https://user-images.githubusercontent.com/76064876/138587380-42fa599a-b1a7-45f4-80d4-d0072ed3b6e3.jpg)

![plate](https://user-images.githubusercontent.com/76064876/138587383-3b0cba1b-6250-46c2-84ef-7430a9d87488.jpg)


# Segmentation
With the arrival of the new license plate image, various filters are first applied to the image and the main location of the license plate is identified using the contour. At this stage, more image processing libraries have been used.
Finally, using conditions such as the height of the numbers or the width of the main location of the license plate numbers, the numbers are separated and entered into the constructed model. Our nerves will be given in the third stage.

![333](https://user-images.githubusercontent.com/76064876/138587641-1d828c11-ca47-43b4-8efd-2ce790a0aa9f.JPG)



# Character recognition (ocr)
At this stage, with the CNN neural network, as well as various filters and activity functions suitable for this network, along with determining the correct values for the hyperparameters, the characters identified in the previous stage will be classified.

![44](https://user-images.githubusercontent.com/76064876/138587695-a56b8750-7238-4a59-9155-f941b752e220.JPG)




# License plate detection program graphics

![گرافیک برنامه](https://user-images.githubusercontent.com/76064876/138587773-b2c062a4-7ac2-4965-96ac-49741004ac98.JPG)
















