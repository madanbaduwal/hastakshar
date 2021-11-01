from django import template
register = template.Library()

@register.filter(name = 'splitByAdtherate')
def splitByAdtherate(value):
    d = value.split('@')
    return d[0]