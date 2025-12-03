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


## Local Navigation 

In the optic of keeping the project as simple as possible, we decided to opt for a potential field algorithm for local navigation. The working principle is to use the distance measurements of the IR sensors, and use them to extract rejection vectors, while the nodes of the path act as attraction points. We then sum these up, using various weights to get the desired behaviour when trying to avoid an obstacle.
<div align="center">

$V = A - \beta\sum_{i=1}^{5}\alpha_iR_i$
</div>

Where A is the attraction vector, $\beta$ is the rejection weight, $\alpha_i$ is the sensor coefficent and $R_i$ the rejection vectors. The weight were determined empirically and the rejection and attraction vectors are calculated as such :  
<div align="center">

$A = d_2\{N, C\}\cdot{}\begin{pmatrix} cos(\alpha) \\ sin(\alpha) \end{pmatrix}$

Where C is the current position of the robot and N the attractor node of the path.

$\alpha = tan^{-1}(\frac{N \times D}{N \cdot D})$

$\alpha$ is the angle between the heading of the robot (D) in global coordinates and the direction to the objectif node (N)

$R_i = \begin{pmatrix} cos(\theta_i)d_i \\ sin(\theta_i)d_i\end{pmatrix}$

$d_i = 12 - \frac{I_i- 1500}{250}$
</div>

Where $d_i$ is the distance of the i-th sensor to the obstacle and $I_i$ is the reading of the i-th infrared sensor. This law was determined empirically based on various testing and calibrations that were made, the distance is given in centimeters. 

Once those parameters are determined, we make the robot go forwards at a constant speed while matching it's heading with the calculated V heading, until the obstacle is cleared. When the robot stops detecting the obstacle, we go into a simple "passing mode" where the robot goes straight for 2 seconds before returning to its normal operating mode and follows back the global path since we consider the obstacle as cleared.

<img src="Report_images/BOMR FSM.png" width=600>

*Decision protocols for the state machine*

 The limitations of this algorithm are quickly met, those being that the creation of local minima blocks the robot in a specific spot as well as not hitting the obstacle again once it was cleared. More over, this relies heavily on setting the right parameters for the various weight and times needed to clear obstacles, so that limits the amount of diverse obstacles we can use. 