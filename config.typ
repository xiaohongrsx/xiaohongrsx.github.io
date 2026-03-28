#import "lib/typ2html/typ2html.typ": *

#let footer-content = [
  2026 \~ Present Shixin Ren 任世鑫
]

#let tag-options = (
  "博客搭建": (preset: "cyan", "icon": "/assets/icons/rocket.svg"),
  "Typst": ("preset": "teal", "icon": "/assets/icons/pen.svg"),
  "写作指南": ("preset": "blue", "icon": "/assets/icons/edit.svg"),
  "配置指南": ("preset": "green", "icon": "/assets/icons/settings.svg"),
  "论文阅读": ("preset": "purple", "icon": "/assets/icons/pen.svg"),
  "LLM": ("preset": "magenta", "icon": "/assets/icons/rocket.svg"),
  "注意力机制": ("preset": "teal", "icon": "/assets/icons/settings.svg"),
  "量化": ("preset": "cyan", "icon": "/assets/icons/settings.svg"),
  "推测解码": ("preset": "blue", "icon": "/assets/icons/rocket.svg"),
)

#let render-tag-link = render-tag-link.with(tag-options: tag-options)
#let render-tag-card = render-tag-card.with(tag-options: tag-options)

#let templates = make-templates(
  site-title: "Ren Shixin's Blog",
  header-links: (
    "/": "首页",
    "/posts/": "文章",
    "/categories/": "分类",
    "/tags/": "标签",
    "/archive/": "归档",
  ),
  title: "Renshixin Blog",
  lang: "zh",
  footer-content: footer-content,
  tag-options: tag-options,
  custom-css: (
    "/assets/custom.css",
  ),
  custom-script: (),
)

#let template-post = templates.post
#let template-page = templates.page
