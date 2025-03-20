---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---

{% if posts == site.categories.["Today I Learn"] and posts == site.categories.["2nd Week"] %}
    {% assign posts = posts %}
    {% for post in posts %}
        {% include archive-single.html type=page.entries_layout %}
    {% endfor %}
{% endif %}