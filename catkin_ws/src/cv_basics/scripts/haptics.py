import math
from temp_hacptic import haptic_controller
import numpy as np
from time import sleep

def haptic_force(x, min_distnace, max_distance, case=2):

    if x < min_distnace:
        return 0
    elif x > max_distance:
        return 0
    if case == 1:
        slope = -1 / (max_distance - min_distnace)
        y  = slope * (x - min_distnace) + 1
        return y
    elif case == 2:
        norm = max_distance - min_distnace
        y = ((max_distance - x) / norm)**2
     
        return y

def directional(angle):
    # input_angle = angle
    input_angle = math.radians(angle)
    right_haptic = 0.5 + (math.sin(input_angle))/2
    left_haptic = 0.5 + (math.sin(-input_angle))/2
    
    return left_haptic, right_haptic

def haptic_feeback(distance, direction, max_dst, min_dst):

    force = haptic_force(distance, min_dst, max_dst)
    left, right = directional(direction)

    print("current_force: %.2f" % force, "right: %.2f "% right, "left: %.2f"%left)
  
   
    right = right * force
    left = left * force
    return left, right


def test_cases(case, max_distance, min_distance):
    
    if case == 1:
        # distance = np.append(np.arange(min_distance, max_distance, 1),(np.arange(max_distance, min_distance, -1)))
        distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance, 1))
        angle = distance * 0
    elif case == 2:
        # distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance, 1))
        distance = np.append(np.arange(min_distance, max_distance, 1),(np.arange(max_distance, min_distance-1, -1)))
        angle  = np.arange(90,270,((270 - 90)//len(distance)))

    elif case == 3:
        distance = np.append(np.arange(min_distance, max_distance, 1),(np.arange(max_distance, min_distance, -1)))
        angle = np.arange(270,450,((450 - 270)//len(distance)))
        angle  = [a if a < 360 else a%360 for a in angle ]  
    
    elif case == 4:
        distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance, 1))
        angle = np.arange(270,450,((450 - 270)//len(distance)))
        angle  = [a if a < 360 else a%360 for a in angle ] 
    else:
        print("not a valid case")

    return distance, angle

def pattern_vibrate(left, right, time=[], count=10):
    curr_count=0
    while curr_count <= count:
        for index,t in enumerate(time):
            if index%2 == 0:
                haptic.vibrate_left(left)
            else:
                haptic.virbate_right(right)
            sleep(t)
        print("current count", curr_count)
        curr_count += 1

# Test near and far haptic

# if __name__ == "__main__":
#     haptic = haptic_controller()
#     min_distance = 5
#     max_distance = 10
#     rand_dst = np.random.randint(min_distance, max_distance,10)
#     print(rand_dst)
#     while True:
#         for x in range(0, len(rand_dst)):
#             force = haptic_force(rand_dst[x], min_distance, max_distance)
#             print(math.floor(force*255), rand_dst[x])
#             haptic.vibrate_both(intensityL=math.floor(force*255), intensityR=math.floor(force*255))    
#             sleep(0.5)
#         break
#     haptic.reset_hacptic()
#     haptic.close()


# directional vibration

# if __name__ == "__main__":
#     haptic = haptic_controller()
#     min_distance = 5
#     max_distance = 10
#     rand_dst = np.random.randint(-max_distance, max_distance,10)
#     print(rand_dst)

#     while True:
#         for x in range(0, len(rand_dst)):
#             force = haptic_force(abs(rand_dst[x]), min_distance, max_distance)
            
#             print(math.floor(force*255), rand_dst[x])

#             if rand_dst[x] < 0:
#                 haptic.virbate_direction('left', [math.floor(force*255)])
#             else:
#                 haptic.virbate_direction('right', [math.floor(force*255)])
#             sleep(0.8)
#         break
#     haptic.reset_hacptic()
#     haptic.close()


if __name__ == "__main__":
    haptic = haptic_controller()
    min_distance = 5
    max_distance = 10
    rand_dst, rand_angle = test_cases(1, max_distance, min_distance)

    print(rand_dst, rand_angle)
    
    scaleL = 50
    scaleR = 255

    while True:
        for x in range(0, len(rand_dst)):
            force = haptic_force(abs(rand_dst[x]), min_distance, max_distance)

            left_haptic, right_hatpic = haptic_feeback(rand_dst[x], rand_angle[x], max_distance, min_distance)
            
            print("current angle", rand_angle[x],"current distance", rand_dst[x],
                   "left haptic ",math.floor(left_haptic*scaleL),
                   "right haptic", math.floor(right_hatpic*scaleR))
            
            if (rand_angle[x] > 160 and rand_angle[x] < 200):
                print("vibrating pattern")
                haptic.vibrate_pattern(left=math.floor(left_haptic*scaleL), right=math.floor(right_hatpic*scaleR), time=[0.3,0.2], count=5)
            
            elif (rand_angle[x] > 340 or rand_angle[x] < 20):
                haptic.vibrate_pattern(left=math.floor(left_haptic*scaleL), right=math.floor(right_hatpic*scaleR), time=[0.1,0.2], count=7)
            
            else:
                haptic.vibrate_both(intensityL=math.floor(left_haptic*scaleL), intensityR=math.floor(right_hatpic*scaleR))
            
            sleep(0.8)
            haptic.reset_hacptic()
            sleep(0.4)
        break
    haptic.reset_hacptic()
    haptic.close()




