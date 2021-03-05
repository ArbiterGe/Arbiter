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
from skimage.transform import resize
# from gym_carla.envs.render import BirdeyeRender
# from gym_carla.envs.route_planner import RoutePlanner
# from gym_carla.envs.misc import *

SHOW_PREVIEW = True
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
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.logger=get_carla_logger()
        self.collision=False
        self.done=False
        self.success=False
        self.action_space=self.get_action_space()
        self.observation_space=self.get_observation_space()
        self.collision_list = []
        self.actor_list = []
        self.lane_invasion_list=[]
        
        self.display_size=256
        self.display_route=True   

        # Once we have a client we can retrieve the world that is currently
        # running.
        self.world = self.client.get_world()

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = self.world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        #print(blueprint_library.filter('vehicle'))
        self.model_3 = blueprint_library.filter('model3')[0]
        #self._init_renderer()
        
    
    

    # def _init_renderer(self):
  
    #      pygame.init()
    #      pygame.font.init()
    #      self.display = pygame.display.set_mode(
    #                    (896,672),
    #                pygame.HWSURFACE | pygame.DOUBLEBUF)

        #  pixels_per_meter = 8
        #  pixels_ahead_vehicle = (IMG_WIDTH) * pixels_per_meter
        #  birdeye_params = {
        #                'screen_size': [self.display_size, self.display_size],
        #                'pixels_per_meter': pixels_per_meter,
        #                'pixels_ahead_vehicle': pixels_ahead_vehicle
        #               }
        #  self.birdeye_render = BirdeyeRender(self.world, birdeye_params)



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
         low=[-1.0,-1.0]
         high=[1.0,1.0]
         return gym.spaces.Box(low=np.array(low),high=np.array(high),dtype=np.float32)

    def destroy_actors(self):
        """

        :return:
        """
      #  import pdb
      #  breakpoint()
        print('destroying actors')
        self.sensor.destroy()
        self.lane_invasion.destroy()
        self.colsensor.destroy()
        self.vehicle.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.actor_list.clear()
        self.lane_invasion_list.clear()
        self.collision_list.clear()
        #print('done.')

    def reset(self):
        if len(self.actor_list)!=0 :
                  self.destroy_actors()            
                  
       

        self.start_pose= random.choice(self.world.get_map().get_spawn_points())
        self.end_pose= random.choice(self.world.get_map().get_spawn_points())
        #self.start_pose= self.world.get_map().get_spawn_points()[2]
      #  self.end_pose= self.world.get_map().get_spawn_points()[20]

        self.vehicle = self.world.spawn_actor(self.model_3, self.start_pose)
        print(f'created {self.vehicle.type_id}')
        self.actor_list.append(self.vehicle)
        self.obs=obs(self.vehicle,self.end_pose)      
        self.collision=False
        self.done=False
        self.success=False
        self.timeout=False
        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.obs.image(data))
        #self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.0))

       # time.sleep(4) # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        lane_invasion=self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion=self.world.spawn_actor(lane_invasion,transform,attach_to=self.vehicle)
        self.actor_list.append(self.lane_invasion)
        self.lane_invasion.listen(lambda event: self.lane_invasion_data(event))
        self.state=None
       
      #  self.routeplanner = RoutePlanner(self.vehicle, 12)
       # self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
      #  self.birdeye_render.set_hero(self.vehicle, self.vehicle.id)
        #self.show()
        while self.obs.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.05))

        return self.obs.convert()

    # def show(self):
    #     #self.surface=rgb_to_display_surface(self.obs.front_camera,self.display_size)
    #     camera=self.obs.front_camera
    #    # camera.transpose()
    
    #     self.surface = pygame.surfarray.make_surface(camera)
    #     self.display.blit(self.surface, (90, 0))
    #     pygame.display.flip()




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
        action=action[0]               
        if action[0]>=0:
            self.control.throttle=min(1,max(0,float(abs(action[0]))))
            self.control.brake=0
        else:
            self.control.throttle=0
            self.control.brake=min(1,max(0,float(abs(action[0]))))
        self.control.steer=min(1,max(-1,float(action[1])))
 


        self.vehicle.apply_control(self.control)
        distance_to_goal=self._get_distance_to_goal(self.vehicle,self.end_pose)
      #  self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
  
        self.success=(distance_to_goal<2.0)
       
                #<2极为成功
     
        if len(self.collision_list)!=0:
            self.collision=True
      #      self.logger.debug("Collision")
    #    import pdb 
      #  pdb.set_trace()
        #self.env_state={'Collision':self.collision , 'success':self.success}
        reward=self.get_reward()
        info={'carla-reward':reward }
        self.done=self.success or self. collision or self.timeout
      #  self.show()
 
        return self.obs.convert(), reward, self.done, info


    def get_reward(self ):
        

        # Distance towards goal (in km)
        # d_x = self.vehicle.get_transform().location.x
        # d_y = self.vehicle.get_transform().location.y
        # d_z = self.vehicle.get_transform().location.z
        # player_location = np.array([d_x, d_y, d_z])
        # goal_location = np.array([self.end_pose.location.x,
        #                           self.end_pose.location.y,
        #                           self.end_pose.location.z])
        d = self.compute_distance(self.vehicle.get_transform().location,self.end_pose.location)    
        # Speed
        v = self.obs.get_forward_speed(self.vehicle) 
        # Collision damage
        c=len(self.collision_list)
        l=len(self.lane_invasion_list)
  

        # Compute reward
        r = 0
        if self.state is not None:
            r += 10 * (self.state['d'] - d)            
            r += 0.05 * (v - self.state['v'])
            r -= 10 *abs( (c - self.state['c']))
            r -= 0.2 *abs( (l - self.state['l']))
        if self.success:
            r+=10000
      #      self.logger.debug("success")

        # Update state
        new_state = {'d': d, 'v': v, 'c': c, 'l': l}
                       
        self.state = new_state

        return float(r)


    def _get_distance_to_goal(self, vehicle, end_pose):

        current_x = vehicle.get_transform().location.x
        current_y = vehicle.get_transform().location.y
        distance_to_goal = np.linalg.norm(np.array([current_x, current_y]) - \
                            np.array([end_pose.location.x, end_pose.location.y]))
        return distance_to_goal





   

