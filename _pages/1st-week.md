---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

{% assign temp_posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn'" %}
{% assign posts = temp_posts | where_exp: "post", "post.categories contains '1st Week'" %}

<h3>📌 Filtered Posts</h3>
<ul>
  {% for post in posts %}
    <li>{{ post.title }} - {{ post.categories | join: ", " }}</li>
  {% else %}
    <li>⚠ No matching posts found.</li>
  {% endfor %}
</ul>