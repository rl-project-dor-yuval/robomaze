# parameters set by the user:
* map from text file
* obeservetion space: 
  - robot_joints_state 
  - robot_location+target_location 
  - distance_from_target (l2)
  - fixed_map 
  - (later) cameras 
  - combinations of them
* rewards for events: 
  - ariving target
  - colision 
  - distance from target
  - walking speed 
  - maybe custom reward from raw data
* end of episode rules
  - game over on colision?
  - timeout (max steps)
* maze size in boxes resulution
  - 5*10
  - 10*10
  - 15*10

action space is a R^8 vector

### in the future:
* custom robot
