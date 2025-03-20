---
title: "Category"
layout: archive
permalink: /categories/
author_profile: true
sidebar_main: true
---

{% assign all_categories = site.posts | map: "categories" | flatten | uniq %}

<h2>📌 Categories</h2>

<ul>
  {% assign grouped_categories = all_categories | sort %}
  
  {% assign parent_categories = "" | split: "," %}
  {% assign sub_categories = "" | split: "," %}
  
  {% for category in grouped_categories %}
    {% assign parts = category | split: "/" %}
    
    {% if parts.size > 1 %}
      {% assign parent = parts[0] %}
      {% assign child = parts[1] %}
      
      {% unless parent_categories contains parent %}
        {% assign parent_categories = parent_categories | push: parent %}
      {% endunless %}
      
      {% unless sub_categories contains category %}
        {% assign sub_categories = sub_categories | push: category %}
      {% endunless %}
    {% else %}
      {% unless parent_categories contains category %}
        {% assign parent_categories = parent_categories | push: category %}
      {% endunless %}
    {% endif %}
  {% endfor %}

  {% for parent in parent_categories %}
    <li><strong>{{ parent }}</strong></li>
    <ul>
      {% for sub in sub_categories %}
        {% if sub contains parent %}
          {% assign sub_parts = sub | split: "/" %}
          <li><a href="{{ site.baseurl }}/categories/{{ sub | slugify }}">{{ sub_parts[1] }}</a></li>
        {% endif %}
      {% endfor %}
    </ul>
  {% endfor %}
</ul>