---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

<h3>📌 Debugging Filtered Posts</h3>
<ul>
  {% for post in posts %}
    <li>{{ post.title }} - {{ post.categories | join: ", " }}</li>
  {% endfor %}
</ul>