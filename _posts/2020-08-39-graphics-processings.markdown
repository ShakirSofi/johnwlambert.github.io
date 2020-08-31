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
```
void draw() {
  line(150, 25, mouseX, mouseY);
}
```
The screen is never cleared, so it becomes clogged with lines. To draw only a single line at a time, we can fill the background inside the `draw()` function:
```
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
