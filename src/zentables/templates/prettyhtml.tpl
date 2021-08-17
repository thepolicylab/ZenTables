{% block before_table %}{% endblock before_table %}
{% block table %}
{% if exclude_styles %}
<table style="border-collapse: collapse; {% if table_local_styles %}{{'; '.join(table_local_styles)|safe}}{% endif %}">
{% else %}
<table id="T_{{uuid}}"{% if table_attributes %} {{table_attributes}}{% endif %} style="border-collapse: collapse; {% if table_local_styles %}{{'; '.join(table_local_styles)|safe}}{% endif %}">
{% endif %}
{% block caption %}
{% if caption and caption is string %}
  <caption>{{caption}}</caption>
{% elif caption and caption is sequence %}
  <caption>{{caption[0]}}</caption>
{% endif %}
{% endblock caption %}
{% block thead %}
<thead>
{% block before_head_rows %}{% endblock %}
{% for r in head %}
{% block head_tr scoped %}
    <tr style="padding: 0; margin: 0;">
{% if exclude_styles %}
{% for c in r %}
{% if c.is_visible != False %}
      <{{c.type}} {{c.attributes}} {% if c.styles %}style="{{'; '.join(c.styles)}}" {% endif %}>{{c.value}}</{{c.type}}>
{% endif %}
{% endfor %}
{% else %}
{% for c in r %}
{% if c.is_visible != False %}
      <{{c.type}} class="{{c.class}}" {{c.attributes}} {% if c.styles %}style="{{'; '.join(c.styles)}}" {% endif %}>{{c.value}}</{{c.type}}>
{% endif %}
{% endfor %}
{% endif %}
    </tr>
{% endblock head_tr %}
{% endfor %}
{% block after_head_rows %}{% endblock %}
</thead>
{% endblock thead %}
{% block tbody %}
<tbody>
{% block before_rows %}{% endblock before_rows %}
{% for r in body %}
{% block tr scoped %}
    <tr style="background-color: white; padding: 0; margin: 0;">
{% if exclude_styles %}
{% for c in r %}{% if c.is_visible != False %}
      <{{c.type}} {{c.attributes}} {% if c.styles %}style="{{'; '.join(c.styles)}}" {% endif %}>{{c.display_value}}</{{c.type}}>
{% endif %}{% endfor %}
{% else %}
{% for c in r %}{% if c.is_visible != False %}
      <{{c.type}} {% if c.id is defined -%} id="T_{{uuid}}{{c.id}}" {%- endif %} class="{{c.class}}" {{c.attributes}} {% if c.styles %}style="{{'; '.join(c.styles)}}" {% endif %}>{{c.display_value}}</{{c.type}}>
{% endif %}{% endfor %}
{% endif %}
    </tr>
{% endblock tr %}
{% endfor %}
{% block after_rows %}{% endblock after_rows %}
</tbody>
{% endblock tbody %}
</table>
{% endblock table %}
{% block after_table %}
{% if show_copy_button and not exclude_styles %}
<input id="B_{{uuid}}" type="button" value="Copy Table" />
<script language="javascript">
document.querySelector("#B_{{uuid}}").addEventListener("click", function () {
  function walkTheDOM(node, func) {
    func(node);
    node = node.firstChild;
    while (node) {
        walkTheDOM(node, func);
        node = node.nextSibling;
    }
  }
  
  function removePadding(node) {
    if (node.tagName === "TH" || node.tagName === "TD"){
      node.style.padding = "0 5px";
    }
  }
  
  let el = document.getElementById("T_{{uuid}}");
  let parent = el.parentNode;
  let elCopy = el.cloneNode(true);
  
  walkTheDOM(elCopy, removePadding);
  
  parent.appendChild(elCopy);

  var body = document.body,
    range,
    sel;
  if (document.createRange && window.getSelection) {
    range = document.createRange();
    sel = window.getSelection();
    sel.removeAllRanges();
    try {
      range.selectNodeContents(elCopy);
      sel.addRange(range);
    } catch (e) {
      range.selectNode(elCopy);
      sel.addRange(range);
    }
  } else if (body.createTextRange) {
    range = body.createTextRange();
    range.moveToElementText(elCopy);
    range.select();
  }
  document.execCommand("copy");
  elCopy.remove();
});

</script>
{% endif %}
{% endblock after_table %}