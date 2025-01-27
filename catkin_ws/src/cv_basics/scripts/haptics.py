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

    elif case ==3:
        y = 1/math.exp((x**2)*math.pi)
        return y

def directional(angle):
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
        distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance+1, 1))
        angle = distance * 0
    elif case == 2:
        # distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance, 1))
        distance = np.append(np.arange(min_distance, max_distance, 1),(np.arange(max_distance, min_distance-1, -1)))
        angle  = np.arange(90,270,((270 - 90)//len(distance)))

    elif case == 3:
        # distance = np.append(np.arange(min_distance, max_distance, 1),(np.arange(max_distance, min_distance, -1)))
        distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance+1, 1))
        angle = np.arange(270,450,((450 - 270)//len(distance)))
        angle  = [a if a < 360 else a%360 for a in angle ]  
        distance = distance * 0
        distance = distance + 5
    elif case == 4:
        distance = np.append(np.arange(max_distance, min_distance, -1), np.arange(min_distance, max_distance+1, 1))
        angle = np.arange(270,450,((450 - 270)//len(distance)-1))
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
                haptic.vibrate_right(right)
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
    max_distance = 20
    rand_dst, rand_angle = test_cases(3, max_distance, min_distance)

    # print(rand_dst, rand_angle)
    
    scaleL = 50
    scaleR = 255

    count = 1

    right_graph = np.array([])
    left_graph = np.array([])

   
    
    x_rms = np.array([])
    y_rms = np.array([])
    z_rms = np.array([])
    while True:
        curr_x = haptic.ds.state.RX
        curr_y = haptic.ds.state.RY

        x_rms = np.append(x_rms,haptic.ds.state.accelerometer.X)
        y_rms = np.append(y_rms, haptic.ds.state.accelerometer.Y)
        z_rms = np.append(z_rms, haptic.ds.state.accelerometer.Z)
        
        # distance = math.sqrt((0- curr_x)**2 + (0-curr_y**2)**2)
        angle =(90+ math.degrees(math.atan2(curr_y, curr_x) +  -1*(np.sign(np.arctan2(curr_y, curr_x))-1) * np.pi))%360
        angle = 0
        force = haptic_force(5, min_distance, max_distance)
        left_haptic, right_hatpic = haptic_feeback(5, angle, max_distance, min_distance)
            
        print("current angle", angle,"current distance", 5,
                   "left haptic ",math.floor(left_haptic*scaleL),
                   "right haptic", math.floor(right_hatpic*scaleR))
        
        if len(x_rms) > 20:
            print("calculating rms")
            rms_x = np.sqrt(np.mean(np.array(x_rms)[-20:]**2))/20
            rms_y = np.sqrt(np.mean(np.array(y_rms)[-20:]**2))/20
            rms_z = np.sqrt(np.mean(np.array(z_rms)[-20:]**2))/20
        

            vibration_value = math.sqrt(rms_x**2 + rms_y**2 + rms_z**2)
            print("vibration value: %.2f" % vibration_value)
        # print("x: %.2f" % rms_x, "y: %.2f" % rms_y, "z: %.2f" % rms_z)
        
        haptic.vibrate_both(intensityL=math.floor(left_haptic*scaleL), intensityR=math.floor(right_hatpic*scaleR))
        sleep(0.8)
        count += 1
        if count%100 == 0:
            print("current count", (count/100)*100, "%")
        if count > 1000:
            break


    # while True:
    #     for x in range(0, len(rand_dst)):
    #         force = haptic_force(abs(rand_dst[x]), min_distance, max_distance)

    #         left_haptic, right_hatpic = haptic_feeback(rand_dst[x], rand_angle[x], max_distance, min_distance)
            
    #         print("current angle", rand_angle[x],"current distance", rand_dst[x],
    #                "left haptic ",math.floor(left_haptic*scaleL),
    #                "right haptic", math.floor(right_hatpic*scaleR))
            
    #         # if (rand_angle[x] > 160 and rand_angle[x] < 200):
    #         #     print("vibrating pattern")
    #         #     haptic.vibrate_pattern(left=math.floor(left_haptic*scaleL), right=math.floor(right_hatpic*scaleR), time=[0.2,0.3], count=5)
            
    #         # elif (rand_angle[x] > 340 or rand_angle[x] < 20):
    #         #     haptic.vibrate_pattern(left=math.floor(left_haptic*scaleL), right=math.floor(right_hatpic*scaleR), time=[0.2,0.3], count=7)
            
    #         # else:
    #         haptic.vibrate_both(intensityL=math.floor(left_haptic*scaleL), intensityR=math.floor(right_hatpic*scaleR))
            
    #         sleep(0.5)
    #     break
    # haptic.reset_hacptic()
    # haptic.close()




