---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'Today I Learn' and post.categories contains '1st Week'" %}

{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}