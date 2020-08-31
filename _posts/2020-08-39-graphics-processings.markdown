---
layout: post
comments: true
permalink: /graphics-processing/
title:  "Graphics applications with Processing"
excerpt: " "
date:   2020-08-30 11:00:00
mathjax: true
---

## Processing
Simple programming environment, with an emphasis on animation and interaction. Processing uses Java-style programming.
(0,0) coordinate is the upper left-hand corner of the display window.

To make a square 400px by 400px display window, and to set the background color to an orange-red (rust) color:

```java
size(400,400);
background(192,64,0); // (R,G,B)
stroke(255); // sets line color to white
stroke(255,0,0); // sets line color to red
```

In Processing, we specify a `setup()` block which runs only once, and a `draw()` block which runs repeatedly. `setup()` can be used for any initialization, e.g. setting the screen size, making the background orange, or setting stroke color to white. The `size()` function must always be the first line inside `setup()`. 
```java
void setup() {
  size(480, 120); // width=480px, height=120px
}
```
The `draw()` block is used to handle animation. Here, we fill the screen with filled circles.
```java
void draw() {
  if (mousePressed) {
    fill(0); // fill color will be black
  } else {
    fill(255); // fill color will be white
  }
  ellipse(mouseX, mouseY, 80, 80); // draws ellipse
}
```

In another variant, we can draw lines that emanate from a single coordinate to wherever the user's mouse cursor is located:
```java
void draw() {
  line(150, 25, mouseX, mouseY);
}
```
The screen is never cleared, so it becomes clogged with lines. To draw only a single line at a time, we can fill the background inside the `draw()` function:
```java
void setup() {
  size(400,400);
  stroke(255);
}

void draw() {
  background(192, 64, 0);
  line(150, 25, mouseX, mouseY);
}
```
Most functions will use the `setup()` and `draw()` blocks. More advnaced mouse handling can also be introduced. In the following example, the `mousePressed()` function will be called whenever the mouse is pressed.
```java
void setup() {
  size(400, 400);
  stroke(255);
}

void draw() {
  line(150, 25, mouseX, mouseY);
}

void mousePressed() {
  background(192, 64, 0);
}
```

If you want to create objects whose size is dependent on the screen, we can use the width and height variables, instead of a number:
```java
ellipse(width/2, height/2, 50, 50);
```
Different renderers may be chosen ,such as P2D, which uses OpenGL for faster rendering of two-dimensional graphics:
```java
size(400, 400, P2D);
```
Images are loaded with the `loadImage()` function, which expect to find a file inside a folder named `data`, which is a subdirectory of the sketch folder.

## Basic Shapes
A few primitive shapes can be specified with `point(x,y)`, `line(x1,y1,x2,y2)`, `rect(x,y,width,height)`, `ellipse(x,y,width,height)`.
```java
point(4,5);
line(1,2,5,2);
rect(1,2,4,3); // top-left corner of rectangle, and then width and height
ellipse(3,3,4,6) // x,y, width=4px, height=6px
```
If instead we draw a rectangle by specifying its centerpoint, we can set `rectMode(CENTER)` before our call to `rect()`. If we prefer to specify a rectangle's shape by two points (top-left and bottom-right corners), we can set `rectMode(CORNERS)`.

## Object-Oriented Programming in Processing

We'll now show how to use object-oriented programming (OOP) in Processing. Suppose we wish to animate the motion of a car (modeled by a rectangle) across the screen. We'll specify 4 attributes: its color, x location, y location, and x speed.

If we didn't employ OOP, we would have a bunch of variables in a main program:
```java
color c = color(0);
float x = 0;
float y = 100;
float speed = 1;
```

In `setup()`, we'll initialize the car color, initialize the car location to the starting point, and initialize car speed.
```java
void setup() {
  size(200,200);
}
```

In `draw()`, we'll fill the background, display car at a specific location with color, and increment the car's location by speed:
```java
void draw() {
  background(255);
  move();
  display();
}

void move() {
  x = x + speed;
  if (x > width) {
    x = 0;
  }
}

void display() {
  fill(c);
  rect(x,y,30,10);
}
```

If we go the OOP way, we can specify a new class -- `Car` -- with the four attributes we discussed before. We will also define its constructor
```java
class Car {
  color c;
  float xpos;
  float ypos;
  float xspeed;
 
  // Constructor
  Car() {
    c = color(255);
    xpos = width/2;
    ypos = height/2;
    xspeed = 1;
  }
 
  void display() {
    rectMode(CENTER);
    fill(c);
    rect(xpos,ypos,20,10);
  }
 
  void drive() {
      xpos = xpos + xspeed;
      if (xpos > width) {
        xpos = 0;
      }
  }
}
```
We'll now use this class we just defined. In the `draw()` method, we'll have access to our `myCar` variable.
```java
// Step 1. Declare an object.
Car myCar;

void setup()  {    
  // Step 2. Initialize object.
  myCar = new Car();  
}

void draw()  {    
  background(255);  
  // Step 3. Call methods on the object.
  myCar.drive();  
  myCar.display();  
}
```

## Curves

## Building Polygons

Rectangle
```java
beginShape();
vertex(30, 20);
vertex(85, 20);
vertex(85, 75);
vertex(30, 75);
endShape(CLOSE);
```
Pentagon

Quad strip:
```java
beginShape(QUAD_STRIP); 
vertex(30, 20); 
vertex(30, 75); 
vertex(50, 20);
vertex(50, 75);
vertex(65, 20); 
vertex(65, 75); 
vertex(85, 20);
vertex(85, 75); 
endShape();
```

## References
1. Ben Fry and Casey Reas. http://processing.org/tutorials/overview
2. Daniel Shiffman. http://processing.org/tutorials/drawing
3. Daniel Shiffman. https://processing.org/tutorials/objects/
4. J. David Eisenberg. http://processing.org/tutorials/curves
