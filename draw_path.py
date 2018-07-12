# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:08:11 2018

@author: FluroSat
"""

line = [path[0]]

for p in path[1:]:
    line.append((p[1],p[2]))

line= [(p1,p0) for p0,p1 in line]

# get an image
base = bw.convert('RGBA')

# make a blank image for the text, initialized to transparent text color
txt = Image.new('RGBA', base.size, (255,255,255,0))

# get a font
fnt = ImageFont.truetype('C:\ProgramData\Anaconda3\Library\lib\DejaVuSerif.ttf', 40)
# get a drawing context
d = ImageDraw.Draw(txt)

# draw text, half opacity
#d.text((10,10), "Hello", font=fnt, fill=(255,255,255,128))
# draw text, full opacity
#d.text((10,60), "World", font=fnt, fill=(255,255,255,255))
d.line(line, fill=(255,0,0,64), width=5)

for p in line:
    r = [p[0]-4, p[1]-4, p[0]+4, p[1]+4]
    d.rectangle(r, fill=(255,0,0,64))
    d.text(p, "{}".format(line.index(p)), font=fnt, fill=(255,0,0,128), align='left')    

out = Image.alpha_composite(base, txt)

out.show()
