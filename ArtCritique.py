import cv2
import imutils
import numpy as np


####
#UTILITY FUNCTIONS
#These functions are chiefly used by the Brain object.
#However, as they do not require the Brain object, they are allowed to exist beyond its scope.
####

def gaussian_activation(data): 
    # This returns a value closer to 1, the closer 'x' is to 0.
    # This holds exactly true for negative OR positive 'x'.
    # e.g., x = -1 and x = 1 theoretically have the same output.
    data = np.abs(data)
    part = -(pow(data, 2))
    return pow(np.e, part)

def linear_activation(data):
    return data

def sinusoidal_activation(data):
    return (np.sin(np.pi * data) + 1) / 2


####
#BRAIN OBJECT
#This is THE neural network. This is the star player.
####

class Brain:
    def __init__(self, complexity = 3):
        self.last_thought = 0;
        self.complexity = complexity

        # Complexity is the most important value here.
        # Complexity determines how many tiles the Brain will split the image into when it analyzes it.
        # Therefore, it also determines how many neurons the brain will have, as each square will have its own neuron.

	# If you're using pre-loaded weights (i.e., if you have a 'weights.txt' file), the complexity is overwritten by what in the weights.txt file.
	# It is EXCESSIVELY important to keep this consistent.
	# DO NOT train a Brain with a complexity of 3 with a training regimen meant for a complexity of anything other than 3.

        self.opinion = 0.5  # Overall fondness of the image. 0 =, 1 = perfect. Unused.
        self.rate = 1.01    # Learning rate: the magnitude that it changes its own weight. Smaller value = more accurate 'thinking', but takes longer to train.
                            # The complexity **SQUARED** is how many tiles the Brain will split the image into, when looking at it.

        print("Initializing...")
        print("Creating memory.")
        self.memory = np.ndarray(shape=(self.complexity, self.complexity), dtype=float, order='F')  # The memory retains the activation function's output for a specific tile.
        print("Initializing weights.")                                                              # Those values in turn make up part of the Brain's over-all image rating.
        self.wgt = np.ndarray(shape=(self.complexity, self.complexity), dtype=float, order='F')     # Each tile has its own neuron, which has its own weight. This creates an array of weights for those neurons.
        for i in range(0, complexity):
            for k in range(0, complexity):
                self.wgt[i, k] = 1.0        # Initialize all weights to 1.0
                #print(self.wgt[i, k])

    def mod_learning(self, modifier): # Unused - but can theoretically be used to modify learning rate on-the-fly.
        self.rate *= modifier

    def new_data(self, dat1, dat2, dat3):
        self.datum_a = dat1 / 255 / 3 # Data can exist as a value between [0, 255].  
                                      # Dividing it by 255 truncates to between [0, 1].
        self.datum_b = dat2 / 255 / 3 # Then, we divide it by 3, so that the *sum* of data is between [0, 1].
          
        self.datum_c = dat3 / 255 / 3
        

    def think(self, i, k):      
        thought = self.wgt[i, k] * (self.datum_a + self.datum_b + self.datum_c)
        self.memory[i, k] = gaussian_activation(thought)

        # This is where the activation function gets called.
        # For a given tile, its hue, saturation, and value is summed.
        # Then, the neuron multiplies it by its weight, and then THAT value gets input into the activation function.
        # Currently, the activation function follows a Gaussian formula, particularly because we want a value between 0 and 1.

    def compare(self, expected, error_margin): # Checks the entire memory array at once for if a given element (a think() function output) is above or below an expected value.
        for i in range(0, self.complexity):
            for k in range(0, self.complexity):
                difference = np.abs(self.memory[i, k] - expected)   # For every tile in memory, analyze the difference between the thought residing at that tile, and our expected value.  
                if (difference > error_margin): # If the difference is too big...
                    if (self.memory[i, k] > expected):  # ...and our expected value is smaller...
                        self.wgt[i, k] = self.wgt[i, k]  / self.rate    # ...shrink the weight!
                        
                    elif (self.memory[i, k]  < expected):   # Likewise, if our expected value is too big...
                        self.wgt[i, k] = self.wgt[i, k]  * self.rate    # ...embiggen the weight!

    def gaussian_compare(self, expected, error_margin):
        # The Gaussian activation function works a little weirdly.
        # For the comparisons... if self.memory[i,k] is too big, we need to make the weights BIGGER. This will make self.memory[i,k] SMALLER.
        # The same holds true opposite: if the thought is too small, we need to make the weight SMALLER.
        # Otherwise, this works exactly the same as the normal compare() function.
        
        for i in range(0, self.complexity):
            for k in range(0, self.complexity):
                difference = np.abs(self.memory[i, k] - expected)

                if (difference > error_margin):
                    if (self.memory[i, k] > expected):
                        self.wgt[i, k] = self.wgt[i, k]  * self.rate
                        
                    elif (self.memory[i, k]  < expected):
                        self.wgt[i, k] = self.wgt[i, k]  / self.rate
                        
    def average_thought(self):
        avg = np.nanmean(self.memory, axis = (0, 1))    # Sum together EVERY thought (a think() function output) in the memory array.
        return avg                                      # Then, average them; return this average as output.                                    
                        

    def look_at_image(self, imagePtr, height, width): 
        for k in range(0, self.complexity):     # For each row in the image...
            for i in range(0, self.complexity): # ...for each column in that row...
                slice_hgt_start = int(k * height / self.complexity)                     # Slice the image into a row: analyze row #k 
                slice_hgt_end = int((slice_hgt_start + height / self.complexity) - 1)   # 

                slice_wdt_start = int((i * width / self.complexity))                    # Pare away the i'th column of the image (producing a square).
                slice_wdt_end = int((slice_wdt_start + width / self.complexity) - 1)    #
                    
                imgslice = imagePtr[slice_hgt_start:slice_hgt_end, slice_wdt_start:slice_wdt_end] # Create an image slice for the parameters we just made (producing a square)

                avg = np.nanmean(imgslice, axis = (0, 1))   # The data of each tile is like its own image.
                hue = avg[0]                                # The data in that image is stored like an array: hue is one axis, saturation another, and value a third
                sat = avg[1]                                # ALL of the pixel data in that tile gets averaged out for each axis...
                val = avg[2]                                #
                self.new_data(hue, sat, val)                # ...then passed to a neuron in the Brain as new_data().
                self.think(i, k)                            # The neuron then thinks about this new data!!!


#####
# Training works like this...
# 1.) Give it an image.
# 2.) Use the 'expected' variable to 'imprint' upon the neuron how it feels about the image.
# 2a.) For example, if we give it an expected value of 1, then we're training the neuron to LOVE the image.
# 2b.) Likewise, if we give it an expected value of 0, then we're training the neuron to LOATHE the image.
# 2c.) A value of 0.5 is... average. The image is just 'okay'.
# 3.) The fondness of the neuron for each square is represented by its output, which is determined by a linear activation function.
# 4.) The fondness of each square is summated, then the average is taken. This is the neuron's overall 'rating' of the image.
# 5.) To make it human digestible, it's then multiplied by 10 to give an 'out-of-ten' score.
    def train(self, image, height, width, cycles, expected = 0.75, error_margin = 0.1, is_gauss = 0):   # Give the Brain an image, its height and width, the # of training cycles, the expected value, the error-margin, and tell it if it's using a Gaussian activation function.
        print("Training...")
        self.look_at_image(image, height, width)    # Analyze the image.
        for i in range(1, cycles):
            print("CYCLE: ", i + 1)                 # Repeat this process 'cycles' times.
            if (is_gauss):
                self.gaussian_compare(expected, error_margin)   # Gaussian activation has to use its own compare() function.
            else:                                               # Read more about it at gaussian_compare()'s definition.
                self.compare(expected, error_margin)            # Anyways... compare what the image analyzes, and if it's not to our liking (according to our expected value and error_margin), tweak weights via compare().

            self.look_at_image(image, height, width)            # Analyze the image again, but this time, using the tweaked weights.
            
    def save_weights(self):             # Save the weights currently loaded to a simple .txt file.
        f = open("weights.txt", "w")    # This is really only used once training concludes.
        f.write(str(self.complexity))   # Do note: you can re-train using a pre-existing weight file.
        for i in range(0, self.complexity):         #... just be careful! If you're using a different activation function, or the complexity is different... you may encounter unwanted results!
            for k in range(0, self.complexity):
                f.write(",")
                f.write(str(self.wgt[i, k]))
            
        f.close()
            
                  
    def load_weights(self):             # If saved weights exist, try to load them here.
        try:                            # To save keypresses (and improve UX), any pre-existing weights will be automatically loaded at runtime.
            f = open("weights.txt", 'r')
            weightdata = f.read()
            
            weightdata = weightdata.split(',')

            # The first integer in weightdata is self.complexity.
            # We need to process it first, because processing the rest of the data is done relative to the value of self.complexity.
            self.complexity = int(weightdata[0])
            size = pow(self.complexity, 2)

            # Now, we can actually process the data!
            a = [0] * (size + 1)
            
            for i in range(0, size + 1):
                a[i] = float(weightdata[i])

            a.pop(0) # Get rid of the complexity datapoint: now, it's ALL weights

            p = 0
            for i in range(0, self.complexity): # Load the weights.
                 for k in range(0, self.complexity):
                    self.wgt[i, k] = a[p] # Copy the value at a[p] to whichever index of self.wgt we're on.
                    p += 1                # We have to use 'p' here because the array storing the weights is 1d, and the actual weight array is 2d.
                                          # We can't use 'i' or 'k' because their ranges are both [0, self.complexity); we need to access data on a[] from [0, self.complexity^2).
        except: # If something goes wrong, this message will output to console.
            print("Couldn't open weights.txt!")
            
        f.close()


            


        
                    
limit_ext = 16    # A pseudo-constant variable, which determines the complexity of the Brain yet-to-come.
exit = 0          # Housekeeping variable, allowing the user to exit whenever when it becomes !0.
image_input_successful = 0 # ^ Similar to above.

brain_a = Brain(limit_ext)  # Generate a new brain, where complexity = limit_ext!
while (exit == 0):          # Until the exit variable is anything BUT zero...
    while (image_input_successful == 0):
        print("Please input the name of the image to be analyzed (including file extension, e.g., 'image1.jpg')...")   # ...ask the user what they want the Brain to analyze.
        x = input()             # ...then receive input.  
        try:
            image = cv2.imread(x)   # Use the OpenCV2 pipeline to store this image as a variable.
            image_input_successful = 1
        except:
            print("ERROR: It seems that your input was wrong! Check for typos, and also check that the image actually exists!")

    print("Will you be training the AI using this image?\n1 -- YES\n0 -- NO")   # Training prompt. 
    do_training = int(input())  # Receive the input to the last question as an integer.

    cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      # Convert the input image into HSV values for the Brain to process.
    (height, width, channels) = image.shape     # Log the dimensions of the image as a set of variables.
    (h, s, v) = image[height - 1,width - 1]     # I think this line of code is redundant and no longer used. But, I don't think it's a good idea to remove it, lest I invite trouble...
    
    brain_a.load_weights()                     # You MUST comment this line out whenever you change limit_ext (which is the default complexity). Then, retrain the model.
    brain_a.look_at_image(image, height, width)# The Brain is instructed to look at the image, analyzing it.

    if do_training: # The training prompt warns the user that they can SERIOUSLY mess up the model if they don't know what they're doing!!!
        print("!!!!WARNING!!!!\nTraining mode selected.\nYOU CAN SCREW UP THE ENTIRE MODEL VERY EASILY BY TRAINING IT WRONG... BE CAREFUL!\nIf you tweak the complexity of the neuron, you *NEED* to retrain it from scratch.\n")
        print("Please input the cycle count...\nWARNING! A high cycle count will take a while! 100~200 is like, 20 seconds!")
        x = int(input()) # You can theoretically do everything right, and if you give it weird values, you can accidentally train it to HATE EVERYTHING.
        print("Please input the expected value (i.e., how much should the AI like this image? 0 = SUCKS, 10 = PERFECT)")
        expected = float(input())   # The Brain thinks an image of value 0.0 is bad, and an image of 1.0 is good.
        brain_a.train(image, height, width, x, expected, 0.02, 1)  # Train the Brain!
        brain_a.save_weights()     # Save the results of the training into a text file.

    print(brain_a.average_thought() * 10) # The average_thought() output is the average output of each neuron in the Brain.
    print("Would you like to exit?\n1 -- YES\n0 -- NO") # If we multiply it by 10, we get the Brain's rating of the image from a scale of 1 to 10.
    exit = int(input())







        
            



