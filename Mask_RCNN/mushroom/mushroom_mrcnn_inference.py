# -*- coding: utf-8 -*-
"""
@author: Alan Wu
"""

import argparse
from mushroom_globalvars import _init, set_value
import mushroom_mrcnn

PNG_FILENAME = '../datasets/mushroom/stage2_air_1_NIR_RT_300.png'

if __name__ == '__main__':
    
    _init()
    
    png_filename = input('PNG image filename [' + PNG_FILENAME + ']: ') or PNG_FILENAME
    
    set_value('args', argparse.Namespace(command = 'splash', weights = 'last', 
                                         image = png_filename,  
                                         dataset = 'None', logs = '..\logs', video = 'None')
             )    
    
    mushroom_mrcnn.cmd_run()
    
    print('Finished.')    