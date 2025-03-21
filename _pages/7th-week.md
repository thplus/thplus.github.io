---
title: "TIL - 7th Week"
layout: archive
permalink: /categories/7th-week/
author_profile: true
sidebar:
    nav: "docs"
---
{% assign posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn'" | where_exp: "post", "post.categories contains '7th Week'" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}