# parameters set by the user:

## Environment Rules Set At Creation:

Each parameter will have default value, (default map, default rewards...) so it will be 
easier to configure only the desired parameters.

* Maze size in boxes resolution (5x5, 10x10, 15x10). maybe more or different sizes, we will see what looks reasonable.
there will be boxes at the boundaries (with normal collision)

* Map from text/csv file (the file will contain the char map where: '#' marks a box for example)

* Initial joints state?
    
* Definition of reward for each event: 
    - arriving target
    - collision 
    - moving speed? 
    - ...
    - maybe we will provide some raw data and let the agent make the reward system

* End of episode rules
    - game over on collision?
    - timeout (max steps)
  
* Observation space:
    - robot joints state (R^20 vector)
    - robot location and target_location (coordinates)
    - distance_from_target (l2)
    - fixed size map (not from camera) 
    - different cameras (tracking camera, static drone from above...)
    - different combinations of the above

In each step, the user can access the observations space and set the R^8 vector controlling the joint motors.

??? should the actions be always a R^8 continuous vector?
 should we define another discrete actions system for algorithms like DQN or should it be done by the agent? ???

### in the future:
* custom robot from URDF instead of the ant
