<div align="center">

# Micro 452 - Basic Of Mobile Robotics
## Project 2025
## group 12 

## The Team 
> Paul Huot-Marchand
> Luca Seelbach
> Manuela Waible
</div>

## Work repartition : 
### Computer Vision :
Attributed to : <b>Paul </b>

### Filtering : 
Attributed to : <b>Manuela</b>
### Global Navigation : 
Attributed to : <b>Luca</b>

### Local Navigation : 
Goal : Using IR sensors, detect for unexpected obstacles and correct trajectory accordingly 
Attributed to : <b>Luca</b>

### Motion : 
Goal : Using diverse inputs from global navigation and filtering to determine motor commands 
Attributed to : <b>Manuela</b>
## 1. Objectifs 
The goal of this project is to use the Thymio robot and a camera to : 
- Create a map of an environment 
- Detect the starting position and target position in the environemnt
- Calculate a global path
- Move the robot towards the target location while avoiding obstacles (permanent and movable) 
- Be insensitive to Kidnapping and camera obscuring 

## 2. Environement : 
For the environment, we chose to have a large white paper sheet with black obstacles to be less sensitive to lighting condition. The only conditions we set on the obstacles where that they are not an isosceles triangle nor round since those shapes are reserved respectively for the robot and the end goal.  We also consider the borders of the sheet as obstacles so that the robot never exists the environement. 
<div align="center">
<img src="Report_images/Environnement.png" style="transform: rotate(-90deg)" width=500px>
</div>


