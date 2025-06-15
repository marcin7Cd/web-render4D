# web-render4D
experimental renderer for 4D objects running on webGPU.

## How 4D is displayed
The 4D object is projected onto a 3D viewspaces and then it's projected onto the screen (look at diagram below). We have two two cameras one in 4D and one in 3D.
![image](https://github.com/user-attachments/assets/8ce67c65-c932-496b-b099-b318e80469e3)

The rendered properly deals with occlusions in 4th dimension. The object, which is lower in 4th dimension can occlude the object placed higher. The 3D cells of 4D object are rendered in such a way that their edges are solid and the cell volume is tineted in the specified color.

## Idea behind the renderer

Instead of calculating projection onto 3D viewspace and then onto screen. I combine them. From each pixel on the screen I draw a "ray" plane that passes through 3 points: the pixel on the screen, the position of the 3d camera, and the position of the 4d camera. Then I draw on this plane the ordinary ray from 3D camera through the pixel on the screen. Then I caluclate all intersections with objects (they are 2D shapes on this plane) and decide, what part was hit by the ordinary ray, when it was projected onto the 3D space (it works, because this projection will be on the ordinary ray thanks to the choice of the ray plane).

![image](https://github.com/user-attachments/assets/a68e7dd0-0826-45cf-99d5-4cf971eb9dc0)
