---
title: "1st Week"
layout: archive
permalink: /categories/1st-week/
author_profile: true
sidebar_main: true
---
{% assign posts = site.categories %}
{% if posts == site.categories.["Today I Learn"] and posts == site.categories.["2nd Week"] %}
    {% for post in posts %}
        {% include archive-single.html type=page.entries_layout %}
    {% endfor %}
{% endif %}