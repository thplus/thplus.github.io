---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

<h3>📌 Debugging Categories</h3>
<ul>
  {% for post in site.posts %}
    <li>{{ post.title }} - {{ post.categories | inspect }}</li>
  {% endfor %}
</ul>