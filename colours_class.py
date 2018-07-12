# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 16:41:21 2018

@author: FluroSat
"""
import re

class Colours:
    def __init__(self):
    
        #https://www.rapidtables.com/web/color/RGB_Color.html
        c = """
         	Red	#FF0000	(255,0,0)
         	Lime	#00FF00	(0,255,0)
         	Blue	#0000FF	(0,0,255)
         	Yellow	#FFFF00	(255,255,0)
         	Cyan / Aqua	#00FFFF	(0,255,255)
         	Magenta / Fuchsia	#FF00FF	(255,0,255)
         	Maroon	#800000	(128,0,0)
         	Olive	#808000	(128,128,0)
         	Green	#008000	(0,128,0)
         	Purple	#800080	(128,0,128)
         	Teal	#008080	(0,128,128)
         	Navy	#000080	(0,0,128)
        """
        
        c = re.findall(r'(\([\d\,]*\))', c)
        self.cList = [eval(x) for x in c]
        self.len = len(self.cList)
        
    def __getitem__(self, i):
        return self.cList[i%self.len]

col = Colours()

print(col[0])