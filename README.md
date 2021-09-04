# License-Plate-Detection
License plate detection using different networks

## I. Introduction
Machine Learning and Artificial Intelligence is very important to this world today, every single technological advances has to do with some machine learning and artificial intelligence including computer vision, natural language processing, autonomus driving, and much more. In this project, I wanted to test computer vision, a part of computer science and improve on it. I choose a very popular concept, license plate detection to carry out my research. First of all, my mentor and I found avery well organized article on computer vision and license plate detection. Then I used this as a framework for my research. 

## II. Research
In order to edit the code, I needed to do more research including looking into the Keras applications and documentation as this was used in the framework. I have also dived deeply in to the reference code and figured out what individual steps are doing.

## III. Project
In order to run the project, I split it into three different sections. The first section is let the computer to be able to detect where the license plate is given a picture. The second step is to recognize where the individual letters are in the lincense plate and then clean them in order to make the computer easier to detect. The last and the most important step is to train the computer using different networks and then detect the letters using machine learning.

### &nbsp;&nbsp;&nbsp; Part 1: Extracting License Plate Location from Image
&nbsp;&nbsp;&nbsp; For this part, I used a pretrained model using wpod net and the following function uses wpod to detect the license plate location with its respective coordinates.

``` Python
def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor
```
The result is

<img src="Images/part1_result (1).jpg">

### &nbsp;&nbsp;&nbsp; Part 2: Recognizing Letters within the License Plate
&nbsp;&nbsp;&nbsp; In this part of the project I had to first greyscale and turn the license plate into a easier formate to use. In order to process the images, we first used this function to make sure our pictures are equal sizes

``` Python
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img
```

Then we use many techniques to get what we want, such as converting the images to 255 scale, then grayscale, then blurring it and also setting a threshhold value to change it to binary and finally, dilate the image. Which is achieved using the following lines

``` Python
if (len(LpImg)): #check if there is at least one license image
      # Scales, calculates absolute values, and converts the result to 8-bit.
      plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
      gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
      blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
      binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
      kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
      thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
```
The result looks like this

<img src="/Images/threshding (1).png">

Lastly, in order to detect the individual letters, I sorted all of the contours of the image and according to its width and height, determine where the letters are. The following segment of code achieves this step.

``` Python
for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 0.5<=ratio<=5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)
```

The result is

<img src="/Images/grab_digit_contour.png">


### &nbsp;&nbsp;&nbsp; Part 3: Training the Computer with different Networks and using them to detect the letters.
&nbsp;&nbsp;&nbsp; Training the computer takes a long time and memory. I first researched what types of networks that I could use with the frameword and I found out that I could use MobileNet, ResNet, and Xception. With the training data provided by Nguyen, I was able to train all of these networks with notable results.

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; MobileNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This is the network that the framework came with. Even though there was a lot of errors and it took me a long time to get it running, I was able to produce some of the results that the original author had. MobileNet uses lightweight deep convolutional neural networks to provide an efficient model for mobile and embedded vision applications. The following function uses the MobileNet structure to start training from the dataset.

``` Python
# Create our model with pre-trained MobileNetV2 architecture from imagenet
def create_model(lr=1e-4,decay=1e-4/25., training=False,output_shape=y.shape[1]):
    baseModel = MobileNetV2(weights="imagenet", 
                            include_top=False,
                            input_tensor=Input(shape=(80, 80, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(output_shape, activation="softmax")(headModel)
    
    model = Model(inputs=baseModel.input, outputs=headModel)
    
    if training:
        # define trainable lalyer
        for layer in baseModel.layers:
            layer.trainable = True
        # compile model
        optimizer = Adam(lr=lr, decay = decay)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=["accuracy"])    
        
    return model
```
The following shows the MobileNet Architecture

<img src="/Images/mobilenet.png">

#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ResNet worked flawlessly within the network but it did took a lot longer than MobileNet on Google Colab. ResNet uses residual learning and builds pyramidal cells so that layers could be skipped. 

The following shows the ResNet Architecture

<img src="/Images/ResNet.png">


#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception:
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Xception took around the same time as ResNet. Xception is a deeper implementation of Inception which uses Deapthwise Separable Convolutions to train the data. 

The following shows the Xception Architecture

<img src="/Images/Xception.png">

### &nbsp;&nbsp;&nbsp; Challenges:
&nbsp;&nbsp;&nbsp; Throughout this whole project, many problems was encountered. First of all, I discovered that Google Colab was really slow in running the models and tests so I decided to use my own computer with the gpu to hopefully make the training go faster. Then I discovered that even with my gpu, the computer still does not run as fast as Google Colab and my memory was running out. In the end, I switched back to Google Colab and tested the rest of the project there. When using the different networks, problems also occured when chosing which version of the librariesare used such as for the keras application and tensorflow. Throughout the process, there was many bugs as well that was discovered. I was able to overcome all the challenges and tested and experienced with machine learning in deapth. 

### &nbsp;&nbsp;&nbsp; Results:


## Conclusion:
Working with programs and code can be very frustrating at times but eventually, everything can be solved. Through this project and research, I know many of the weakness and flaw of the original framework. I can do more research and find a better solution to extract the license plates more effectively and also can improve on letter detection as well as the networks.

## References
