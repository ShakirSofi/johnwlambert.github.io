---
layout: post
comments: true
permalink: /graphics-processing/
title:  "Graphics applications with Processing"
excerpt: " "
date:   2020-08-30 11:00:00
mathjax: true
---

```java
void setup() {
  size(480, 120); // width=480px, height=120px
}

void draw() {
  if (mousePressed) {
    fill(0);
  } else {
    fill(255);
  }
  ellipse(mouseX, mouseY, 80, 80); // draws ellipse
}

```
