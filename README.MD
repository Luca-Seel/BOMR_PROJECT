# Basic Of Mobile Robotics (BOMR) project 2025 group 12 

## The Team 
> Paul Huot-Marchand
> Romain Klein
> Manuela Waible
> Luca Seelbach

## Objectifs 
The goal of this project is to use the Thymio robot and a camera to : 
- Map an environment 
- Detect starting position and target position in the environemnt
- Calculate a global trajectory 
- Move the robot towards target location while avoiding obstacles (permanent and movable) 
- Be insensitive to Kidnapping and camera obscuring 

## Modules : 
###1. Computer Vision :
Goal : Take the image from the camera and determine the map of the environment, the start position of the robot and the goal. 
Attributed to : Paul 

###2. Filtering : 
Goal : Take the various sensors input (IMU, motor rotation), and the camera image to input into an bayesian filter and keep a statistical position of the robot position.
Attributed to : Manuela

###3. Global Navigation : 
Goal : From the environment map, starting position and goal position, determine the fastest path 
Attributed to : Luca

###4. Local Navigation : 
Goal : Using IR sensors, detect for unexpected obstacles and correct trajectory accordingly 
Attributed to : Luca

###5. Motion : 
Goal : Using diverse inputs from global & local navigation determine motor commands 

