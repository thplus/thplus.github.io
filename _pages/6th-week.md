---
title: "TIL - 6th Week"
layout: archive
permalink: /categories/6th-week/
author_profile: true
sidebar:
    nav: "docs"
---
{% assign posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn'" | where_exp: "post", "post.categories contains '6th Week'" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}