from grasp_generator import GraspGenerator
from environment.utilities import Camera
from environment.env import Environment
from utils import YcbObjects, summarize
import pybullet as p
import numpy as np
import sys
import random
import os
import random
sys.path.append('network')

def isolated_obj_scenario(n):
    vis = True
    output = True
    debug = True

    objects = YcbObjects('objects/ycb_objects', 'results', n, ['ChipsCan', 'MustardBottle', 'TomatoSoupCan'])
    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x, center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug, num_objs=0, gripper_type='140')
    generator = GraspGenerator(network_path, camera, 5)

    # objects.obj_names = ['MustardBottle', 'ChipsCan', 'TomatoSoupCan']

    for obj_name in objects.obj_names:
        print(obj_name)       
        for _ in range(n):

            special_case = objects.check_special_case(obj_name)
            env.load_isolated_obj(objects.get_obj_path(obj_name), special_case)
            env.move_away_arm()
            
            rgb, depth, _ = camera.get_cam_img()
            grasps, save_name = generator.predict_grasp(rgb, depth, n_grasps=3, show_output=output)
            for grasp in grasps:
                objects.add_try(obj_name)
                x, y, z, roll, opening_len, obj_height = grasp
                # print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}, obj height:{obj_height}')
                if vis:
                    debugID = p.addUserDebugLine([x, y, z], [x, y, 1.2], [0, 0, 1])
                
                succes_grasp, succes_target = env.grasp((x, y, z), roll, opening_len, obj_height)
                # print(f'Grasped:{succes_grasp} Target:{succes_target}')                
                if vis:
                    p.removeUserDebugItem(debugID)              
                if succes_grasp:
                    objects.add_succes_grasp(obj_name)
                if succes_target:
                    objects.add_succes_target(obj_name)
                    if save_name is not None:
                        os.rename(save_name + '.png', save_name + '_SUCCESS.png')
                    break
                env.reset_obj()
            env.remove_obj()

    objects.write_json()
    summarize(objects.save_dir, n)

def pile_scenario(n):
    vis = False
    output = True
    debug = False

    objects = YcbObjects('objects/ycb_objects', 'results', n, ['ChipsCan', 'MustardBottle', 'TomatoSoupCan'])
    center_x, center_y = 0.05, -0.52
    network_path = 'network/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
    camera = Camera((center_x, center_y, 1.9), (center_x, center_y, 0.785), 0.2, 2.0, (224, 224), 40)
    env = Environment(camera, vis=vis, debug=debug, num_objs=0, gripper_type='140', finger_length=0.06)
    generator = GraspGenerator(network_path, camera, 5)

    objects.obj_names = ['TennisBall','PottedMeatCan', 'Banana', 'ChipsCan']

    # objects.shuffle_objects()
    paths = []

    for obj_name in objects.obj_names:
        paths.append(objects.get_obj_path(obj_name))  
    env.create_pile(paths)

    tries = 0

    fails = 0
    while len(env.obj_ids) != 0 and fails < 3:

        env.move_away_arm()
        rgb, depth, _ = camera.get_cam_img()
        grasps, save_name = generator.predict_grasp(rgb, depth, n_grasps=3, show_output=output)
        fails = 0

        for grasp in grasps:
            tries += 1
            objects.add_try(obj_name)
            x, y, z, roll, opening_len, obj_height = grasp
            # print(f'x:{x} y:{y}, z:{z}, roll:{roll}, opening len:{opening_len}, obj height:{obj_height}')
            if vis:
                debugID = p.addUserDebugLine([x, y, z], [x, y, 1.2], [0, 0, 1])
            
            succes_grasp, succes_target = env.grasp((x, y, z), roll, opening_len, obj_height)
            print(f'Grasp={succes_grasp} Target={succes_target}')
            if not succes_target:
                env.reset_all_obj()
                fails += 1
            if succes_target:
                break
            if fails == 3:
                break

    print(f'Tries={tries}')
    print(f'Perc cleared={(4 - len(env.obj_ids)) / 4}')
    print(f'Acc={(4 - len(env.obj_ids)) / tries}')

if __name__ == '__main__':
    # isolated_obj_scenario(1)
    pile_scenario(1)