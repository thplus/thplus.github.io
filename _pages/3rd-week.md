---
title: "TIL - 3rd Week"
layout: archive
permalink: /categories/3rd-week/
author_profile: true
sidebar_main: true
---
{% assign posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn'" | where_exp: "post", "post.categories contains '3rd Week'" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}