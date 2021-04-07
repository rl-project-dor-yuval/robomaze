## API specifications

# parameters set by the user:

* Initially set the episode rules:
    1. Maze size in boxes resolution (5*10, 10*10, 15*10)
        * map from text file
       (The joints state is initialized by default values.
        
    2. Rewards for events: 
      - arriving target
      - collision 
      - distance from target
      - walking speed 
      - maybe custom reward from raw data

    3. End of episode rules
      - game over on colision?
      - timeout (max steps)

In every single episode , user can access the observations space.
and control the simulation using the R8 vector controlling the joint motors.

* obeservetion space: 
  - robot_joints_state (R8 vector?)
  - robot_location+target_location 
  - distance_from_target (l2)
  - fixed_map 
  - combinations of them
  - (later) cameras 


### in the future:
* custom robot
