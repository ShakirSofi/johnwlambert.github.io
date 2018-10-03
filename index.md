---
layout: default
---

|![](/images/mom_melissa_huntington_small.jpg) | Ph.D. Candidate<br>[School of Interactive Computing](https://www.ic.gatech.edu/)<br>[Georgia Institute of Technology](https://www.gatech.edu/) | 

## Bio

I am a Ph.D. student at Georgia Tech, advised by Professor [James Hays](https://www.cc.gatech.edu/~hays/). I completed my Bachelor's and Master's degrees in Computer Science at Stanford University in 2018, specializing in artificial intelligence.  My research interests revolve around the semantic understanding of geometric data.

I enjoy creating teaching materials for topics in computer vision, a field which relies heavily upon numerical optimization and statistical machine learning tools.

You can reach me at johnlambert AT gatech DOT edu. Some of my code can be [found here](http://github.com/johnwlambert/).

Office: 
College of Computing Building (CCB) 
Room 309 (3rd Floor)
Georgia Institute of Technology
801 Atlantic Drive NW,
Atlanta, GA 30332

[[My CV]](/assets/cv.pdf)


## Research
Humans have an amazing ability to understand the world through their visual system, but computers struggle with the task. The real world is 3D, not 2D, so reasoning in the 2D image plane is insufficient. 3D is high-dimensional and challenging and has a high data requirement.

Today, the computer vision community is learning how to use statistical learning and numerical optimization techniques for handling 3D sensor data. Accurate 3D understanding of environments will have enormous benefit for people all over the world, with implications for safer transportation and safer workplaces.

## Teaching
<div class="home">
  
  <ul class="posts">
    {% for post in site.posts %}
      <li>
        <span class="post-date">{{ post.date | date: "%b %-d, %Y" }}</span>
        <a class="post-link" href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
        <br>
        {{ post.excerpt }}
      </li>
    {% endfor %}
  </ul>

</div>