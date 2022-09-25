# -*- coding: utf-8 -*-

def _init():
    global vars
    vars = {}
    
def set_value(key, value):
    global vars
    vars[key] = value
    
def get_value(key):
    return(vars[key])