---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

{% if posts == site.categories.["Today I Learn"] and posts == site.categories.["1st Week"] %}
    {% assign posts = posts %}
{% endif %}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}