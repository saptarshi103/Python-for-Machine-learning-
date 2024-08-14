# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:35:31 2024

@author: sapta
"""

class Car:
    def __init__(self):
        self.acc=False
        self.cluch=False
        self.brk=False
    def start(self):
        self.cluch=True
        self.acc=True
        print("Car started...")
Car1=Car()
Car1.start()