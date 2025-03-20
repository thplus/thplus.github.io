---
title: "Category"
layout: archive
permalink: /categories/
author_profile: true
sidebar_main: true
---

<h2>📌 Categories</h2>

<ul>
  {% assign category_tree = "" | split: "," %}

  {% for post in site.posts %}
    {% for category in post.categories %}
      {% unless category_tree contains category %}
        {% assign category_tree = category_tree | push: category %}
      {% endunless %}
    {% endfor %}
  {% endfor %}

  {% assign category_tree = category_tree | sort %}

  {% assign parent_categories = "" | split: "," %}
  {% assign sub_categories = "" | split: "," %}

  {% for category in category_tree %}
    {% assign parent = category[0] %}
    {% assign child = category[1] %}

    {% unless parent_categories contains parent %}
      {% assign parent_categories = parent_categories | push: parent %}
    {% endunless %}

    {% if child %}
      {% assign sub_categories = sub_categories | push: category %}
    {% endif %}
  {% endfor %}

  {% for parent in parent_categories %}
    <li><strong>{{ parent }}</strong></li>
    <ul>
      {% for sub in sub_categories %}
        {% assign sub_parts = sub | split: " " %}
        {% if sub_parts[0] == parent %}
          <li>
            <a href="{{ site.baseurl }}/categories/{{ sub_parts[1] | slugify }}">{{ sub_parts[1] }}</a>
          </li>
        {% endif %}
      {% endfor %}
    </ul>
  {% endfor %}
</ul>
