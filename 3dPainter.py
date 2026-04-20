from ursina import *
from ursina.shaders import lit_with_shadows_shader
import cv2
import numpy as np
import random
import math
from PIL import Image
import HandTrackingBase as htb

app = Ursina()
window.borderless = False
window.size = (1920, 1080)

capture = cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

detector = htb.handDetector(trackCon=.75, maxHands=1)

# creating the 3d space in AR, and lock it to camera view
video_bg = Entity( #error
    model='quad',
    texture='white_cube',
    parent=camera,
    z=50, #push it to back 
    scale=(camera.aspect_ratio * 37, 37)
)

#make a 3d anchor
draw_anchor = Entity(position=(0,0,0))
#trackers
current_stroke = None
prev_pos = None

sun = DirectionalLight(y=2, z=3, shadow_map_resolution=(1028,1028))
sun.look_at(Vec3(0,0,0))
rVal = random.randint(1, 255)
gVal = random.randint(1, 255)
bVal = random.randint(1, 255)

def bake_stroke(stroke_to_bake):
    if stroke_to_bake:
        stroke_to_bake.combine()
        stroke_to_bake.shader = lit_with_shadows_shader

def update():
    global current_stroke, prev_pos
    #read from camera
    success, image = capture.read()
    if not success:
        return 
    image = cv2.flip(image, 1)

    #convert the opencv bgr to ursina rgb and put it on background
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    #scaling background
    true_ratio = w/h
    video_bg.scale = (36.4 * true_ratio, 36.4)

    #must convert numpy array to a PIL image so it can be read by ursina
    pil_image = Image.fromarray(image_rgb)
    video_bg.texture = Texture(pil_image)

    #track
    image = detector.findHands(image, draw=False)
    lmList, _ = detector.findPosition(image, draw=False)


    if len(lmList) >= 21:
        fingers = detector.fingersUp(lmList)
        #get finger coords
        x, y = lmList[8][1], lmList[8][2]
        raw_depth = detector.findDepth(lmList)

        world_z = np.interp(raw_depth, [0, 350], [50, -10]) #bring z foward

        visible_h = world_z*0.73 #default FOV
        visible_w = visible_h * camera.aspect_ratio
        #coord mapping, according to john gpt, opencv is 0-1280, ursina is -7 to 7, so must translate
        world_x = ((x/w)-0.5) * visible_w #change sensitiviy
        world_y = -((y/h)-0.5) * visible_h
        #map raw pixels to ursina z space, when raw depth is 50, Z becomes 5 (pushed into screen), and when its 250, z becomes -5 (pulled away)
        current_pos = Vec3(world_x, world_y, world_z)
        #print(f"Raw Hand Width: {raw_depth} | Calculated Z: {world_z}")

        #draw
        if fingers[1] == 1 and fingers[2] == 0:
            #start new stroke
            if current_stroke is None:
                current_stroke = Entity(parent=draw_anchor)
                prev_pos = current_pos

            #linear interpretation, draws filler spheres so it doesnt look dotted
            distance = (current_pos - prev_pos).length()
            steps = int(distance / 0.5) + 1

            for i in range(steps):
                linInter_pos = lerp(prev_pos, current_pos, i / steps)

            #make sphere where fingertip is
                Entity(
                    model = 'sphere',
                    color=color.rgba32(r=rVal, g=gVal, b=bVal, a=255), #not sure if random is a ursina feature
                    scale=0.5,
                    position = linInter_pos,
                    parent = current_stroke,
                    shader = lit_with_shadows_shader
                )
            prev_pos = current_pos

        #if entering selection mode
        else:
            if current_stroke is not None:
                #add a ~3 second wait time before combining 
                invoke(bake_stroke, current_stroke, delay=3)
                current_stroke = None
                prev_pos = None

def input(key):
    global current_stroke
    if key == "e":
        if current_stroke is not None:
            current_stroke.combine()
            current_stroke.shader = lit_with_shadows_shader
            current_stroke = None
        exportObj(draw_anchor)


def exportObj(anchor, filename = "drawing.obj"):
    print("Exporting...")
    with open(filename, 'w') as file:
        offset = 1
        for stroke in anchor.children:
            if not stroke.model:
                continue
            #write all 3d points
            for v in stroke.model.vertices:
                world_x = v[0] + stroke.x
                world_y = v[1] + stroke.y
                world_z = v[2] + stroke.z
                file.write(f"v {world_x} {world_y} {world_z}\n")
            #write connections
            for t in stroke.model.triangles:
                file.write(f"f {t[0]+offset} {t[1]+offset} {t[2]+offset}\n")
            #update offset so next stroke's connections are correct
            offset += len(stroke.model.vertices)

    print("Export complete")


app.run()



