import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import subprocess
import random
from cv2 import cv2 as cv
import gym
import numpy as np
import time
#from observation_utils import CameraException
import math
from carla_logger import get_carla_logger


SHOW_PREVIEW = False
IMG_WIDTH=640
IMG_HEIGHT=640
SECONDS_PER_EPISODE = 10


class obs:
    SHOW_CAM = SHOW_PREVIEW
    def __init__(self,vehicle,end_pose):
        self.vehicle=vehicle
        self.end_pose=end_pose
        self.im_width=IMG_WIDTH
        self.im_height=IMG_HEIGHT

    def vector(self):
        self.start_pose=self.vehicle.get_transform()
        self.v = np.array([
               self.start_pose.location.x,
               self.start_pose.location.y,
               self.start_pose.rotation.yaw,
               self.get_forward_speed(self.vehicle) ,
               self.vehicle.get_acceleration().x,      
               self.vehicle.get_acceleration().y,        
               self.end_pose.location.x,
               self.end_pose.location.y,
               self.end_pose.rotation.yaw
                ])
        return  self.v

    def image(self,img):          
            i = np.array(img.raw_data)
             #np.save("iout.npy", i)
            i2 = i.reshape((self.im_height, self.im_width, 4))
            i3 = i2[:, :, :3]
            i4 = i3[:, :, ::-1]
            if self.SHOW_CAM:
                cv.imshow("",i4)
                cv.waitKey(1)
            self.front_camera = i4.copy()

    def convert(self):
            return {'img':self.front_camera.copy(),'v':self.vector()}

    def get_forward_speed(self,vehicle):
            vel = vehicle.get_velocity()
            return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)



class CarEnv:
    
    STEER_AMT = 1.0

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []

    front_camera = None
    collision_list = []
    lane_invasion_list=[]

    def __init__(self):
        self.client = carla.Client('localhost',3000)
        self.client.set_timeout(5.0)
        self.logger=get_carla_logger()
        self.collision=False
        self.done=False
        self.success=False
        self.action_space=self.get_action_space()
        self.observation_space=self.get_observation_space()
        self.collision_list = []
        self.actor_list = []
        self.waypoint_list=[]
        self.lane_invasion_list=[]    
        self.world = self.client.load_world('Town10HD')
        self.map=self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter('model3')[0]
    
    
    def get_observation_space(self):
        self.im_width=IMG_WIDTH
        self.im_height=IMG_HEIGHT
        self.chennel=3
        self.vbounds = np.array([
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 30],
                [-100, 100],
                [-100, 100],
                [0, 100], # TODO: Fine-tune
                [0, 100], # TODO: Fine-tune
                [0, 100]
            ])  
        img_shape = (self.chennel, self.im_height, self.im_width)
        img_box = gym.spaces.Box(low=0, high=1, shape=img_shape, dtype=np.float32)
        v_low = self.vbounds[:, 0]
        v_high = self.vbounds[:, 1]
        v_box = gym.spaces.Box(low=v_low, high=v_high, dtype=np.float32)       
        d = {'img': img_box, 'v': v_box}
        return gym.spaces.Dict(d)

    def  get_action_space(self):        
         low=[-0.2,-0.1]
         high=[0.35,0.1]
         return gym.spaces.Box(low=np.array(low),high=np.array(high),dtype=np.float32)

    def destroy_actors(self):
        """

        :return:
        """
        print('destroying actors')
        self.sensor.destroy()
        self.lane_invasion.destroy()
        self.colsensor.destroy()
        self.vehicle.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.actor_list.clear()
        self.lane_invasion_list.clear()
        self.collision_list.clear()
        self.waypoint_list.clear()
        #print('done.')

    def reset(self):
        if len(self.actor_list)!=0 :
                  self.destroy_actors()            
                  
       

        #self.start_pose= random.choice(self.world.get_map().get_spawn_points())
      #  self.end_pose= random.choice(self.world.get_map().get_spawn_points())
        self.start_pose= self.world.get_map().get_spawn_points()[10]
        self.end_pose= self.world.get_map().get_spawn_points()[6]
        # import pdb
        # breakpoint()
        self.vehicle = self.world.spawn_actor(self.model_3, self.start_pose)
        print(f'created {self.vehicle.type_id}')
        self.actor_list.append(self.vehicle)
        self.obs=obs(self.vehicle,self.end_pose)      
        self.collision=False
        self.done=False
        self.success=False
        self.timeout=False
        self.waypoint=self.map.get_waypoint(self.vehicle.get_transform().location)      
     
        #camera
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.obs.image(data))
        #colsensor
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        #lane
        lane_invasion=self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion=self.world.spawn_actor(lane_invasion,transform,attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion)
        self.lane_invasion.listen(lambda event: self.lane_invasion_data(event))
        self.state=None

        while self.obs.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.1))
        return self.obs.convert()

        
    def compute_distance(self, location_1, location_2):
         x = location_2.x - location_1.x
         y = location_2.y - location_1.y
         z = location_2.z - location_1.z
         norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
         return norm


    def collision_data(self, event):
        self.collision_list.append(event)

    def lane_invasion_data(self, event):
          self.lane_invasion_list.append(event)


       



    def step(self, action):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        self.control=carla.VehicleControl()        
       # action=action[0]               
        if action[0]>=0:
            self.control.throttle=min(1,max(0,float(abs(action[0]))))
            self.control.brake=0
        else:
            self.control.throttle=0
            self.control.brake=min(1,max(0,float(abs(action[0]))))
        self.control.steer=min(1,max(-1,float(action[1])))
        self.vehicle.apply_control(self.control)     
        self.success=self.is_within_distance_ahead(self.vehicle.get_transform(),self.end_pose,2) 
        self.waypoint=self.map.get_waypoint(self.vehicle.get_transform().location)      
      
                #<2极为成功     

        if len(self.collision_list)!=0:
            self.collision=True
            self.logger.debug("Collision")
        self.waypoint_list.append(self.waypoint)
        self.draw_waypoints(self.world,self.waypoint_list)        
        #self.env_state={'Collision':self.collision , 'success':self.success}
        reward=self.get_reward()
        info={'carla-reward':reward}
        self.done=self.success or self. collision or self.timeout

        return self.obs.convert(), reward, self.done, info


    def get_reward(self ):

        yaw=abs(self.waypoint.transform.rotation.yaw-self.vehicle.get_transform().rotation.yaw)
        #print('yaw:',yaw)
        distance_road=self.distance_vehicle(self.waypoint, self.vehicle.get_transform())

        d = self.compute_distance(self.vehicle.get_transform().location,self.end_pose.location)
        
        # # Speed
        # self.next_waypoint=self.waypoint.next(0.1)[0]
        # location1=self.waypoint.transform.location
        # location2=self.next_waypoint.transform.location
        # roll=math.atan((location1.x-location2.x)/(location1.y-location2.y))
        # print('roll=',roll)
        v = self.obs.get_forward_speed(self.vehicle) 
     #   print (f"distance={d},Speed={v}")

        #print('vehicle_speed:___',v)
        # Collision damage
        c=len(self.collision_list)
        l=len(self.lane_invasion_list)
  

        # Compute reward
        r = 0.00
        if self.state is not None:
            if self.state['d']>d:
               r += 5.5 * (self.state['d'] - d)
            else:
                r += 0.9* (self.state['d'] - d)

            r += 0.0875* v           
            r -= 1 *abs( (c - self.state['c']))
           # print(r)   
            r -= 0.5 *abs( (l - self.state['l']))
            r -=0.01*yaw
           # r -=0.5*self.state['road_d']
            #print (f"distance_diff=%3f{%3(self.state['d'] - d)},Speed_diff={v},Line_diff={l - self.state['l']},Coll_diff={c - self.state['c']}，Yaw_diff={yaw}")
          
        if self.success:
            r+=10000
      #      self.logger.debug("success")

        # Update state
        new_state = {'d': d, 'v': v, 'c': c, 'l': l, 'road_d':distance_road,'yaw':yaw}
                       
        self.state = new_state

        return float('%.3f' %r) 

    def draw_waypoints(self,world, waypoints, z=0.5):
               """
                       Draw a list of waypoints at a certain height given in z.

                        :param world: carla.world object
                     :param waypoints: list or iterable container with the waypoints to draw
                 :param z: height in meters
                   """
               for wpt in waypoints:
                     wpt_t = wpt.transform
                     begin = wpt_t.location + carla.Location(z=z)
                     angle = math.radians(wpt_t.rotation.yaw)
                     end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                     world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
   
    def distance_vehicle(self, waypoint, vehicle_transform):
                  """
                        Returns the 2D distance from a waypoint to a vehicle

                          :param waypoint: actual waypoint
                       :param vehicle_transform: transform of the target vehicle
                          """
                  loc = vehicle_transform.location
                  x = waypoint.transform.location.x - loc.x
                  y = waypoint.transform.location.y - loc.y

                  return math.sqrt(x * x + y * y)

    # def _get_directions(self, current_point, end_point):

    #     directions = self._planner.get_next_command(
    #         (current_point.location.x,
    #          current_point.location.y, 0.22),
    #         (current_point.orientation.x,
    #          current_point.orientation.y,
    #          current_point.orientation.z),
    #         (end_point.location.x, end_point.location.y, 0.22),
    #         (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
    #     return directions



    def is_within_distance(self, target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
            """
                           Check if a target object is within a certain distance from a reference object.
                        A vehicle in front would be something around 0 deg, while one behind around 180 deg.

                         :param target_location: location of the target object
                         :param current_location: location of the reference object
                         :param orientation: orientation of the reference object
                         :param max_distance: maximum allowed distance
                         :param d_angle_th_up: upper thereshold for angle
                         :param d_angle_th_low: low thereshold for angle (optional, default is 0)
                         :return: True if target object is within max_distance ahead of the reference object
            """
            target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
            norm_target = np.linalg.norm(target_vector)

                 # If the vector is too short, we can simply stop here
            if norm_target < 0.001:
                  return True

            if norm_target > max_distance:
                  return False

            forward_vector = np.array(
                   [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
            d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

            return d_angle_th_low < d_angle < d_angle_th_up


    def is_within_distance_ahead(self,target_transform, current_transform, max_distance):
            """
                              Check if a target object is within a certain distance in front of a reference object.

                              :param target_transform: location of the target object
                              :param current_transform: location of the reference object
                               :param orientation: orientation of the reference object
                               :param max_distance: maximum allowed distance
                              :return: True if target object is within max_distance ahead of the reference object
            """
            target_vector = np.array([target_transform.location.x - current_transform.location.x, target_transform.location.y - current_transform.location.y])
            norm_target = np.linalg.norm(target_vector)

            # If the vector is too short, we can simply stop here
            if norm_target < 0.001:
                        return True

            if norm_target > max_distance:
                        return False

            fwd = current_transform.get_forward_vector()
            forward_vector = np.array([fwd.x, fwd.y])
            d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

            return d_angle < 90.0


   

