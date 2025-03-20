---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

{% assign posts = site.categories["1st Week"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}