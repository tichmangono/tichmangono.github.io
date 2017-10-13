---
layout: page
title: Python | Data Science | Public Health
tagline:
---
{% include JB/setup %}

I use this space to share what I learn about Python and Machine Learning and to showcase 
personal projects and interests. 

## About me

I am an aspiring Machine Learning Engineer/Data Scientist. I apply analytics to develop 
strategies for HIV prevention and advise ministries of health across southern Africa 
on high-impact public health decisions. Although my career has spanned several fields - 
economics, investment banking, private equity and lately, public health - my ability 
to solve problems is the unifying theme. Whether I am developing an SQL-backed, ASP.NET 
web application to increase retail promotions’ profitability; developing evidence-based 
targets for HIV prevention; or using machine learning to predict HIV supply chain 
stock-outs, I rely on a combination of logic, technology and data to tackle issues that 
I care about.

    
## Articles
<ul class="posts">
  {% for post in site.posts %}
    <li><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>