---
title: "Ajou University Project"
layout: archive
permalink: /categories/ajou-university/
author_profile: true
sidebar:
    nav: "docs"
---

{% assign posts = site.categories.["Ajou University"] %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}