---
layout: page
title: Machine Learning | Data Science | Public Health
tagline: Machine Learning Musings
---
{% include JB/setup %}

This is a personal blog where I share lessons about what I learn about Python, Machine Learning, Public Health, and Data Science
I also showcase several projects that I have worked on. 

## About ME

Aspiring Machine Learning Engineering/Data Scientist.

    
## Current Posts
<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>