---
title: "TIL - 4th Week"
layout: archive
permalink: /categories/4th-week/
author_profile: true
sidebar:
    nav: "docs"
---
{% assign posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn'" | where_exp: "post", "post.categories contains '4th Week'" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}